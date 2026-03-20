import argparse
import json
import os
import time

import numpy as np

try:
    import mlx.core as mx
    import mlx.optimizers as optim
except ImportError as exc:
    raise ImportError(
        "This script requires MLX. Install it first, for example: pip install mlx"
    ) from exc

import gpt_peft_mlx as base


def gpt_hidden_states(input_ids, attn_mask, params, n_head, n_layer):
    _, seq_len = input_ids.shape
    tok = params["wte.weight"][input_ids]
    pos = params["wpe.weight"][mx.arange(seq_len)]
    x = tok + pos.reshape(1, seq_len, -1)

    for layer in range(n_layer):
        x = base.transformer_block(x, attn_mask, base.block_view(params, layer), n_head)

    return base.layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])


def response_token_mask(input_ids, prompt_lengths, pad_token_id):
    mask = np.zeros(input_ids.shape, dtype=np.float32)
    for row, prompt_len in enumerate(prompt_lengths):
        start = min(int(prompt_len), input_ids.shape[1])
        valid = input_ids[row] != pad_token_id
        mask[row, start:] = valid[start:].astype(np.float32)
    return mx.array(mask)


def mean_pool_hidden(hidden, token_mask):
    masked = hidden * token_mask[:, :, None]
    denom = mx.maximum(mx.sum(token_mask, axis=1, keepdims=True), 1.0)
    return mx.sum(masked, axis=1) / denom


def init_reward_model(hidden_size, seed=0):
    rng = np.random.RandomState(seed)
    return {
        "weight": mx.array((rng.randn(hidden_size, 1) * 0.02).astype(np.float32)),
        "bias": mx.zeros((1,), dtype=mx.float32),
    }


def reward_forward(hidden, token_mask, reward_params):
    pooled = mean_pool_hidden(hidden, token_mask)
    return (pooled @ reward_params["weight"]).reshape(-1) + reward_params["bias"][0]


def pad_sequences(sequences, pad_value, block_size):
    batch = np.full((len(sequences), block_size), pad_value, dtype=np.int64)
    attn = np.zeros((len(sequences), block_size), dtype=np.float32)
    for row, seq in enumerate(sequences):
        length = min(len(seq), block_size)
        batch[row, :length] = seq[:length]
        attn[row, :length] = 1.0
    return mx.array(batch), mx.array(attn)


def sequence_logprob(logits, target_ids, token_mask):
    shifted = logits[:, :-1, :]
    targets = target_ids[:, 1:]
    mask = token_mask[:, 1:]
    log_probs = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
    return mx.sum(picked * mask, axis=-1)


def collect_rollout(model_params, tokenizer, hf_config, prompt, block_size, max_new_tokens, temperature, top_k):
    prefix = f"Instruction:\n{prompt.strip()}\n\nResponse:\n"
    prefix_ids = tokenizer.encode(prefix)
    generated = list(prefix_ids)

    for _ in range(max_new_tokens):
        window = generated[-block_size:]
        input_ids = mx.array(np.array([window], dtype=np.int64))
        attn_mask = mx.ones((1, len(window)), dtype=mx.float32)
        logits = base.gpt_forward(
            input_ids,
            attn_mask,
            model_params,
            hf_config["n_head"],
            hf_config["n_layer"],
        )
        next_id = base.top_k_sample(logits[0, -1], top_k=top_k, temperature=temperature)
        generated.append(next_id)
        if next_id == tokenizer.pad_token_id:
            break

    response_ids = generated[len(prefix_ids) :]
    return {
        "prompt": prompt,
        "prefix": prefix,
        "prefix_ids": prefix_ids,
        "response_ids": response_ids,
        "response_text": tokenizer.decode(response_ids),
        "full_ids": generated,
    }


def collect_preferences(args, params, tokenizer, hf_config, block_size):
    if args.prompts_file is None:
        raise ValueError("--prompts-file is required in collect mode")

    prompts = []
    with open(args.prompts_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                prompts.append(json.loads(line)["prompt"])
            else:
                prompts.append(line)

    if not prompts:
        raise ValueError(f"no prompts found in {args.prompts_file}")

    os.makedirs(os.path.dirname(args.preference_data) or ".", exist_ok=True)
    saved = 0
    for idx, prompt in enumerate(prompts, start=1):
        print(f"\nprompt {idx}/{len(prompts)}")
        print(prompt)
        left = collect_rollout(
            params,
            tokenizer,
            hf_config,
            prompt,
            block_size,
            args.sample_tokens,
            args.temperature,
            args.top_k,
        )
        right = collect_rollout(
            params,
            tokenizer,
            hf_config,
            prompt,
            block_size,
            args.sample_tokens,
            args.temperature,
            args.top_k,
        )

        print("\n[A]")
        print(left["response_text"])
        print("\n[B]")
        print(right["response_text"])
        choice = input("\nChoose better response: [a/b/s/q] ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice not in {"a", "b"}:
            print("skipping invalid choice")
            continue

        chosen = left if choice == "a" else right
        rejected = right if choice == "a" else left
        row = {
            "prompt": prompt,
            "chosen": chosen["response_text"],
            "rejected": rejected["response_text"],
        }
        with open(args.preference_data, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        saved += 1
        print(f"saved preference {saved} -> {args.preference_data}")

    print(f"\nfinished preference collection with {saved} saved pairs")


def load_preference_dataset(path, tokenizer, block_size):
    dataset = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["prompt"].strip()
            prefix = f"Instruction:\n{prompt}\n\nResponse:\n"
            chosen_ids = tokenizer.encode(prefix + row["chosen"].strip())[:block_size]
            rejected_ids = tokenizer.encode(prefix + row["rejected"].strip())[:block_size]
            prefix_ids = tokenizer.encode(prefix)[:block_size]
            dataset.append(
                {
                    "prompt": prompt,
                    "prompt_len": len(prefix_ids),
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
            )
    if not dataset:
        raise ValueError(f"no preference examples found in {path}")
    return dataset


def reward_loss_fn(reward_params, batch, params, hf_config, pad_token_id):
    chosen_ids, chosen_attn = pad_sequences(batch["chosen_ids"], pad_token_id, batch["block_size"])
    rejected_ids, rejected_attn = pad_sequences(batch["rejected_ids"], pad_token_id, batch["block_size"])

    chosen_mask = response_token_mask(np.array(chosen_ids), batch["prompt_lens"], pad_token_id)
    rejected_mask = response_token_mask(np.array(rejected_ids), batch["prompt_lens"], pad_token_id)

    chosen_hidden = gpt_hidden_states(chosen_ids, chosen_attn, params, hf_config["n_head"], hf_config["n_layer"])
    rejected_hidden = gpt_hidden_states(rejected_ids, rejected_attn, params, hf_config["n_head"], hf_config["n_layer"])

    chosen_reward = reward_forward(chosen_hidden, chosen_mask, reward_params)
    rejected_reward = reward_forward(rejected_hidden, rejected_mask, reward_params)
    preference_logits = chosen_reward - rejected_reward
    return mx.mean(mx.logaddexp(mx.zeros_like(preference_logits), -preference_logits))


def make_preference_batch(dataset, batch_size):
    indices = np.random.randint(0, len(dataset), size=batch_size)
    return {
        "chosen_ids": [dataset[idx]["chosen_ids"] for idx in indices],
        "rejected_ids": [dataset[idx]["rejected_ids"] for idx in indices],
        "prompt_lens": [dataset[idx]["prompt_len"] for idx in indices],
        "block_size": max(
            max(len(dataset[idx]["chosen_ids"]), len(dataset[idx]["rejected_ids"])) for idx in indices
        ),
    }


def save_reward_model(reward_params, path):
    np.savez(path, weight=np.array(reward_params["weight"]), bias=np.array(reward_params["bias"]))


def load_reward_model(path):
    payload = np.load(path)
    return {
        "weight": mx.array(payload["weight"].astype(np.float32)),
        "bias": mx.array(payload["bias"].astype(np.float32)),
    }


def train_reward_model(args, reward_params, pref_dataset, params, hf_config, tokenizer):
    optimizer = optim.Adam(learning_rate=args.reward_lr, betas=[0.9, 0.999], eps=1e-8)
    loss_and_grad = mx.value_and_grad(reward_loss_fn)
    start = time.time()

    for step in range(1, args.reward_steps + 1):
        batch = make_preference_batch(pref_dataset, args.reward_batch_size)
        loss, grads = loss_and_grad(reward_params, batch, params, hf_config, tokenizer.pad_token_id)
        optimizer.update(reward_params, grads)
        mx.eval(reward_params, optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.reward_steps:
            elapsed = time.time() - start
            print(f"reward step {step:04d}/{args.reward_steps} | loss {loss.item():.4f} | {elapsed:.1f}s")


def rollout_logprobs_and_reward(trainable_params, prompt_batch, params, reference_params, reward_params, tokenizer, hf_config, args):
    merged_params = base.merge_params(params, trainable_params)
    sequences = []
    prompt_lens = []
    for prompt in prompt_batch:
        rollout = collect_rollout(
            merged_params,
            tokenizer,
            hf_config,
            prompt,
            args.block_size,
            args.sample_tokens,
            args.temperature,
            args.top_k,
        )
        sequences.append(rollout["full_ids"])
        prompt_lens.append(len(rollout["prefix_ids"]))

    input_ids, attn_mask = pad_sequences(sequences, tokenizer.pad_token_id, args.block_size)
    token_mask = response_token_mask(np.array(input_ids), prompt_lens, tokenizer.pad_token_id)

    current_logits = base.gpt_forward(input_ids, attn_mask, merged_params, hf_config["n_head"], hf_config["n_layer"])
    ref_logits = base.gpt_forward(input_ids, attn_mask, reference_params, hf_config["n_head"], hf_config["n_layer"])
    hidden = gpt_hidden_states(input_ids, attn_mask, merged_params, hf_config["n_head"], hf_config["n_layer"])
    rewards = reward_forward(hidden, token_mask, reward_params)
    current_logp = sequence_logprob(current_logits, input_ids, token_mask)
    ref_logp = sequence_logprob(ref_logits, input_ids, token_mask)
    return rewards, current_logp, ref_logp


def rl_loss_fn(trainable_params, prompt_batch, params, reference_params, reward_params, tokenizer, hf_config, args):
    rewards, current_logp, ref_logp = rollout_logprobs_and_reward(
        trainable_params,
        prompt_batch,
        params,
        reference_params,
        reward_params,
        tokenizer,
        hf_config,
        args,
    )
    advantages = rewards - mx.mean(rewards)
    advantages = advantages / (mx.std(advantages) + 1e-6)
    kl = current_logp - ref_logp
    objective = advantages * current_logp - args.kl_coef * kl
    return -mx.mean(objective)


def sample_prompt_batch(pref_dataset, batch_size):
    indices = np.random.randint(0, len(pref_dataset), size=batch_size)
    return [pref_dataset[idx]["prompt"] for idx in indices]


def run_rlhf(args, params, hf_config, tokenizer):
    model_block_size = hf_config["n_ctx"]
    if args.block_size is None:
        args.block_size = model_block_size
    else:
        args.block_size = min(args.block_size, model_block_size)

    pref_dataset = load_preference_dataset(args.preference_data, tokenizer, args.block_size)
    reward_params = (
        load_reward_model(args.load_reward)
        if args.load_reward is not None
        else init_reward_model(hf_config["n_embd"], seed=args.seed)
    )

    if args.reward_steps > 0:
        print(f"loaded {len(pref_dataset)} preference pairs")
        train_reward_model(args, reward_params, pref_dataset, params, hf_config, tokenizer)
        if args.save_reward is not None:
            save_reward_model(reward_params, args.save_reward)
            print(f"saved reward model to {args.save_reward}")

    reference_params = dict(params)
    trainable_params = base.extract_trainable_params(params)
    optimizer = optim.Adam(learning_rate=args.lr, betas=[0.9, 0.999], eps=1e-8)
    loss_and_grad = mx.value_and_grad(rl_loss_fn)
    start = time.time()

    for step in range(1, args.rl_steps + 1):
        prompt_batch = sample_prompt_batch(pref_dataset, args.batch_size)
        loss, grads = loss_and_grad(
            trainable_params,
            prompt_batch,
            params,
            reference_params,
            reward_params,
            tokenizer,
            hf_config,
            args,
        )
        optimizer.update(trainable_params, grads)
        params.update(trainable_params)
        mx.eval(trainable_params, optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.rl_steps:
            elapsed = time.time() - start
            print(f"rl step {step:04d}/{args.rl_steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    if args.save_lora is not None:
        base.save_lora(params, hf_config, args.save_lora)
        print(f"saved RLHF LoRA adapters to {args.save_lora}")


def main():
    parser = argparse.ArgumentParser(description="Toy RLHF on top of gpt_peft_mlx.py using preferences, a tiny reward model, and PPO-lite LoRA updates.")
    parser.add_argument("--mode", choices=["collect", "train", "infer"], default="train")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rl-steps", type=int, default=50)
    parser.add_argument("--reward-steps", type=int, default=100)
    parser.add_argument("--reward-batch-size", type=int, default=4)
    parser.add_argument("--reward-lr", type=float, default=1e-3)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--sample-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--preference-data", type=str, default="preferences.jsonl")
    parser.add_argument("--prompts-file", type=str, default=None)
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--save-lora", type=str, default="mlx_gpt2_rlhf_lora.npz")
    parser.add_argument("--load-reward", type=str, default=None)
    parser.add_argument("--save-reward", type=str, default="mlx_reward_model.npz")
    parser.add_argument("--sample-prompt", type=str, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)

    params, _, hf_config = base.init_model_from_hf(args.model, rank=args.rank, alpha=args.alpha)
    hf_config["lora_rank"] = args.rank
    hf_config["lora_alpha"] = args.alpha
    tokenizer = base.GPT2Tokenizer()

    if args.load_lora is not None:
        loaded = base.load_lora(params, args.load_lora)
        print(f"loaded {loaded} LoRA tensors from {args.load_lora}")

    if args.mode == "collect":
        model_block_size = hf_config["n_ctx"]
        block_size = model_block_size if args.block_size is None else min(args.block_size, model_block_size)
        collect_preferences(args, params, tokenizer, hf_config, block_size)
        return

    if args.mode == "train":
        run_rlhf(args, params, hf_config, tokenizer)

    prompt = base.resolve_prompt(args)
    block_size = hf_config["n_ctx"] if args.block_size is None else min(args.block_size, hf_config["n_ctx"])
    print("\nsample:")
    text = base.sample_text(
        prompt,
        tokenizer,
        params,
        hf_config["n_head"],
        hf_config["n_layer"],
        block_size,
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stream=True,
    )
    if not text.startswith(prompt):
        print(text)


if __name__ == "__main__":
    main()
