import argparse
import json
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


def pad_sequences(sequences, pad_value, block_size):
    batch = np.full((len(sequences), block_size), pad_value, dtype=np.int64)
    attn = np.zeros((len(sequences), block_size), dtype=np.float32)
    for row, seq in enumerate(sequences):
        length = min(len(seq), block_size)
        batch[row, :length] = seq[:length]
        attn[row, :length] = 1.0
    return mx.array(batch), mx.array(attn)


def response_token_mask(input_ids, prompt_lengths, pad_token_id):
    mask = np.zeros(input_ids.shape, dtype=np.float32)
    for row, prompt_len in enumerate(prompt_lengths):
        start = min(int(prompt_len), input_ids.shape[1])
        valid = input_ids[row] != pad_token_id
        mask[row, start:] = valid[start:].astype(np.float32)
    return mx.array(mask)


def sequence_logprob(logits, target_ids, token_mask):
    shifted = logits[:, :-1, :]
    targets = target_ids[:, 1:]
    mask = token_mask[:, 1:]
    log_probs = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
    picked = mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
    return mx.sum(picked * mask, axis=-1)


def load_preference_dataset(path, tokenizer, block_size):
    dataset = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["prompt"].strip()
            prefix = f"Instruction:\n{prompt}\n\nResponse:\n"
            prefix_ids = tokenizer.encode(prefix)[:block_size]
            chosen_ids = tokenizer.encode(prefix + row["chosen"].strip())[:block_size]
            rejected_ids = tokenizer.encode(prefix + row["rejected"].strip())[:block_size]
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


def make_preference_batch(dataset, batch_size):
    indices = np.random.randint(0, len(dataset), size=batch_size)
    return {
        "prompts": [dataset[idx]["prompt"] for idx in indices],
        "chosen_ids": [dataset[idx]["chosen_ids"] for idx in indices],
        "rejected_ids": [dataset[idx]["rejected_ids"] for idx in indices],
        "prompt_lens": [dataset[idx]["prompt_len"] for idx in indices],
        "block_size": max(
            max(len(dataset[idx]["chosen_ids"]), len(dataset[idx]["rejected_ids"])) for idx in indices
        ),
    }


def batch_logprobs(model_params, sequences, prompt_lens, tokenizer, hf_config, block_size):
    input_ids, attn_mask = pad_sequences(sequences, tokenizer.pad_token_id, block_size)
    token_mask = response_token_mask(np.array(input_ids), prompt_lens, tokenizer.pad_token_id)
    logits = base.gpt_forward(
        input_ids,
        attn_mask,
        model_params,
        hf_config["n_head"],
        hf_config["n_layer"],
    )
    return sequence_logprob(logits, input_ids, token_mask)


def dpo_loss_fn(trainable_params, batch, params, reference_params, tokenizer, hf_config, beta):
    policy_params = base.merge_params(params, trainable_params)
    block_size = batch["block_size"]

    policy_chosen = batch_logprobs(
        policy_params,
        batch["chosen_ids"],
        batch["prompt_lens"],
        tokenizer,
        hf_config,
        block_size,
    )
    policy_rejected = batch_logprobs(
        policy_params,
        batch["rejected_ids"],
        batch["prompt_lens"],
        tokenizer,
        hf_config,
        block_size,
    )
    ref_chosen = batch_logprobs(
        reference_params,
        batch["chosen_ids"],
        batch["prompt_lens"],
        tokenizer,
        hf_config,
        block_size,
    )
    ref_rejected = batch_logprobs(
        reference_params,
        batch["rejected_ids"],
        batch["prompt_lens"],
        tokenizer,
        hf_config,
        block_size,
    )

    chosen_advantage = policy_chosen - ref_chosen
    rejected_advantage = policy_rejected - ref_rejected
    preference_margin = beta * (chosen_advantage - rejected_advantage)
    return mx.mean(mx.logaddexp(mx.zeros_like(preference_margin), -preference_margin))


def run_dpo(args, params, hf_config, tokenizer):
    model_block_size = hf_config["n_ctx"]
    block_size = model_block_size if args.block_size is None else min(args.block_size, model_block_size)
    dataset = load_preference_dataset(args.preference_data, tokenizer, block_size)
    print(f"loaded {len(dataset)} preference pairs")

    reference_params = dict(params)
    trainable_params = base.extract_trainable_params(params)
    optimizer = optim.Adam(
        learning_rate=args.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
    )
    loss_and_grad = mx.value_and_grad(dpo_loss_fn)
    start = time.time()

    for step in range(1, args.steps + 1):
        batch = make_preference_batch(dataset, args.batch_size)
        loss, grads = loss_and_grad(
            trainable_params,
            batch,
            params,
            reference_params,
            tokenizer,
            hf_config,
            args.beta,
        )
        optimizer.update(trainable_params, grads)
        params.update(trainable_params)
        mx.eval(trainable_params, optimizer.state, loss)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - start
            print(f"dpo step {step:04d}/{args.steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    if args.save_lora is not None:
        base.save_lora(params, hf_config, args.save_lora)
        print(f"saved DPO LoRA adapters to {args.save_lora}")


def main():
    parser = argparse.ArgumentParser(description="Toy DPO example on top of gpt_peft_mlx.py using chosen/rejected preference pairs.")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--preference-data", type=str, default="preferences.jsonl")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-lora", type=str, default="mlx_gpt2_dpo_lora.npz")
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    np.random.seed(args.seed)

    params, _, hf_config = base.init_model_from_hf(args.model, rank=args.rank, alpha=args.alpha)
    hf_config["lora_rank"] = args.rank
    hf_config["lora_alpha"] = args.alpha
    tokenizer = base.GPT2Tokenizer()

    if args.load_lora is not None:
        loaded = base.load_lora(params, args.load_lora)
        print(f"loaded {loaded} LoRA tensors from {args.load_lora}")

    if args.mode == "train":
        run_dpo(args, params, hf_config, tokenizer)

    block_size = hf_config["n_ctx"] if args.block_size is None else min(args.block_size, hf_config["n_ctx"])
    prompt = base.resolve_prompt(args)
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
