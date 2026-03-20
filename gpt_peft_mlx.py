import argparse
import json
import os
import time
from glob import glob

import numpy as np

try:
    import mlx.core as mx
    import mlx.optimizers as optim
except ImportError as exc:
    raise ImportError(
        "This script requires MLX. Install it first, for example: pip install mlx"
    ) from exc


def softmax(x, axis=-1):
    shifted = x - mx.max(x, axis=axis, keepdims=True)
    exps = mx.exp(shifted)
    return exps / mx.sum(exps, axis=axis, keepdims=True)


def gelu(x):
    coeff = np.float32(np.sqrt(2.0 / np.pi))
    inner = x + 0.044715 * (x**3)
    return 0.5 * x * (1.0 + mx.tanh(coeff * inner))


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    inv_std = 1.0 / mx.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    return x_hat * gamma + beta


def cross_entropy(logits, targets, ignore_index=-100):
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if not bool(np.array(mx.any(valid))):
        raise ValueError("batch has no valid targets")

    safe_targets = mx.where(valid, flat_targets, 0)
    shifted = flat_logits - mx.max(flat_logits, axis=-1, keepdims=True)
    log_probs = shifted - mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True))
    picked = mx.take_along_axis(log_probs, safe_targets.reshape(-1, 1), axis=-1).reshape(-1)
    valid_f = valid.astype(log_probs.dtype)
    return -(mx.sum(picked * valid_f) / mx.sum(valid_f))


def lora_linear(x, weight, bias, lora_a, lora_b, alpha):
    base = x @ weight + bias
    rank = lora_a.shape[1]
    scale = alpha / rank
    delta = (x @ lora_a) @ lora_b
    return base + delta * scale


def causal_self_attention(x, attn_mask, block, n_head):
    batch_size, seq_len, n_embd = x.shape
    head_dim = n_embd // n_head

    qkv = lora_linear(
        x,
        block["attn.c_attn.weight"],
        block["attn.c_attn.bias"],
        block["attn.c_attn.lora_a"],
        block["attn.c_attn.lora_b"],
        block["attn.c_attn.alpha"],
    )
    q = qkv[:, :, :n_embd]
    k = qkv[:, :, n_embd : 2 * n_embd]
    v = qkv[:, :, 2 * n_embd :]

    q = q.reshape(batch_size, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)

    scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / np.sqrt(head_dim))
    causal = mx.triu(mx.ones((seq_len, seq_len), dtype=mx.float32), k=1) * -1e9
    scores = scores + causal.reshape(1, 1, seq_len, seq_len)

    if attn_mask is not None:
        key_mask = (1.0 - attn_mask.astype(mx.float32)).reshape(batch_size, 1, 1, seq_len) * -1e9
        scores = scores + key_mask

    weights = softmax(scores, axis=-1)
    out = weights @ v
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, n_embd)

    return lora_linear(
        out,
        block["attn.c_proj.weight"],
        block["attn.c_proj.bias"],
        block["attn.c_proj.lora_a"],
        block["attn.c_proj.lora_b"],
        block["attn.c_proj.alpha"],
    )


def mlp(x, block):
    hidden = gelu(
        lora_linear(
            x,
            block["mlp.c_fc.weight"],
            block["mlp.c_fc.bias"],
            block["mlp.c_fc.lora_a"],
            block["mlp.c_fc.lora_b"],
            block["mlp.c_fc.alpha"],
        )
    )
    return lora_linear(
        hidden,
        block["mlp.c_proj.weight"],
        block["mlp.c_proj.bias"],
        block["mlp.c_proj.lora_a"],
        block["mlp.c_proj.lora_b"],
        block["mlp.c_proj.alpha"],
    )


def transformer_block(x, attn_mask, block, n_head):
    x = x + causal_self_attention(layer_norm(x, block["ln_1.weight"], block["ln_1.bias"]), attn_mask, block, n_head)
    x = x + mlp(layer_norm(x, block["ln_2.weight"], block["ln_2.bias"]), block)
    return x


def init_lora_pair(in_dim, out_dim, rank):
    return {
        "lora_a": mx.array(np.random.randn(in_dim, rank).astype(np.float32) * 0.01),
        "lora_b": mx.zeros((rank, out_dim), dtype=mx.float32),
    }


def hf_weight_key_names(layer):
    prefix = f"h.{layer}."
    return {
        prefix + "ln_1.weight",
        prefix + "ln_1.bias",
        prefix + "attn.c_attn.weight",
        prefix + "attn.c_attn.bias",
        prefix + "attn.c_proj.weight",
        prefix + "attn.c_proj.bias",
        prefix + "ln_2.weight",
        prefix + "ln_2.bias",
        prefix + "mlp.c_fc.weight",
        prefix + "mlp.c_fc.bias",
        prefix + "mlp.c_proj.weight",
        prefix + "mlp.c_proj.bias",
    }


def maybe_unwrap_state_dict(state):
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if "model" in state and isinstance(state["model"], dict):
        return state["model"]
    return state


def resolve_key_name(state, *candidates):
    for key in candidates:
        if key in state:
            return key
    raise KeyError(f"None of these keys were found: {candidates}")


def load_safetensors_file(path):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "This script requires safetensors for torch-free checkpoint loading. "
            "Install it with: pip install safetensors"
        ) from exc

    state = {}
    with safe_open(path, framework="np") as handle:
        for key in handle.keys():
            state[key] = handle.get_tensor(key)
    return state


def load_safetensors_state(model_dir):
    single_file_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_file_path):
        return load_safetensors_file(single_file_path)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index_data = json.load(handle)

        weight_map = index_data.get("weight_map", {})
        if not weight_map:
            raise ValueError(f"No weight_map found in {index_path}")

        state = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = os.path.join(model_dir, shard_name)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
            state.update(load_safetensors_file(shard_path))
        return state

    candidates = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if len(candidates) == 1:
        return load_safetensors_file(candidates[0])
    if candidates:
        state = {}
        for candidate in candidates:
            state.update(load_safetensors_file(candidate))
        return state

    bin_candidates = sorted(glob(os.path.join(model_dir, "*.bin")))
    if bin_candidates:
        raise FileNotFoundError(
            "Found PyTorch .bin checkpoint files, but this script no longer depends on torch "
            "and only supports safetensors checkpoints. Download model.safetensors "
            "(or safetensor shards plus model.safetensors.index.json) instead."
        )

    raise FileNotFoundError(
        f"Could not find a safetensors checkpoint in {model_dir}. "
        "Expected model.safetensors or sharded safetensors files."
    )


def resolve_model_dir(model_name):
    if os.path.isdir(model_name):
        return model_name

    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_name,
        allow_patterns=["config.json", "*.safetensors", "*.index.json"],
    )


def init_model_from_hf(model_name="gpt2", rank=4, alpha=8.0):
    model_dir = resolve_model_dir(model_name)

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    state = maybe_unwrap_state_dict(load_safetensors_state(model_dir))

    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]

    params = {}
    lora_params = []

    wte_key = resolve_key_name(
        state,
        "transformer.wte.weight",
        "wte.weight",
        "model.transformer.wte.weight",
    )
    wpe_key = resolve_key_name(
        state,
        "transformer.wpe.weight",
        "wpe.weight",
        "model.transformer.wpe.weight",
    )
    ln_f_weight_key = resolve_key_name(
        state,
        "transformer.ln_f.weight",
        "ln_f.weight",
        "model.transformer.ln_f.weight",
    )
    ln_f_bias_key = resolve_key_name(
        state,
        "transformer.ln_f.bias",
        "ln_f.bias",
        "model.transformer.ln_f.bias",
    )

    params["wte.weight"] = mx.array(state[wte_key].astype(np.float32))
    params["wpe.weight"] = mx.array(state[wpe_key].astype(np.float32))
    params["ln_f.weight"] = mx.array(state[ln_f_weight_key].astype(np.float32))
    params["ln_f.bias"] = mx.array(state[ln_f_bias_key].astype(np.float32))
    params["lm_head.weight"] = mx.array(state[wte_key].astype(np.float32).T)

    for layer in range(n_layer):
        prefix = f"h.{layer}."
        for name in hf_weight_key_names(layer):
            hf_name = resolve_key_name(
                state,
                "transformer." + name,
                name,
                "model.transformer." + name,
            )
            params[name] = mx.array(state[hf_name].astype(np.float32))

        module_shapes = {
            prefix + "attn.c_attn": (n_embd, 3 * n_embd),
            prefix + "attn.c_proj": (n_embd, n_embd),
            prefix + "mlp.c_fc": (n_embd, 4 * n_embd),
            prefix + "mlp.c_proj": (4 * n_embd, n_embd),
        }
        for module_name, (in_dim, out_dim) in module_shapes.items():
            pair = init_lora_pair(in_dim, out_dim, rank)
            params[module_name + ".lora_a"] = pair["lora_a"]
            params[module_name + ".lora_b"] = pair["lora_b"]
            params[module_name + ".alpha"] = np.float32(alpha)
            lora_params.extend([module_name + ".lora_a", module_name + ".lora_b"])

    return params, lora_params, config


def block_view(params, layer):
    prefix = f"h.{layer}."
    return {
        "ln_1.weight": params[prefix + "ln_1.weight"],
        "ln_1.bias": params[prefix + "ln_1.bias"],
        "attn.c_attn.weight": params[prefix + "attn.c_attn.weight"],
        "attn.c_attn.bias": params[prefix + "attn.c_attn.bias"],
        "attn.c_attn.lora_a": params[prefix + "attn.c_attn.lora_a"],
        "attn.c_attn.lora_b": params[prefix + "attn.c_attn.lora_b"],
        "attn.c_attn.alpha": params[prefix + "attn.c_attn.alpha"],
        "attn.c_proj.weight": params[prefix + "attn.c_proj.weight"],
        "attn.c_proj.bias": params[prefix + "attn.c_proj.bias"],
        "attn.c_proj.lora_a": params[prefix + "attn.c_proj.lora_a"],
        "attn.c_proj.lora_b": params[prefix + "attn.c_proj.lora_b"],
        "attn.c_proj.alpha": params[prefix + "attn.c_proj.alpha"],
        "ln_2.weight": params[prefix + "ln_2.weight"],
        "ln_2.bias": params[prefix + "ln_2.bias"],
        "mlp.c_fc.weight": params[prefix + "mlp.c_fc.weight"],
        "mlp.c_fc.bias": params[prefix + "mlp.c_fc.bias"],
        "mlp.c_fc.lora_a": params[prefix + "mlp.c_fc.lora_a"],
        "mlp.c_fc.lora_b": params[prefix + "mlp.c_fc.lora_b"],
        "mlp.c_fc.alpha": params[prefix + "mlp.c_fc.alpha"],
        "mlp.c_proj.weight": params[prefix + "mlp.c_proj.weight"],
        "mlp.c_proj.bias": params[prefix + "mlp.c_proj.bias"],
        "mlp.c_proj.lora_a": params[prefix + "mlp.c_proj.lora_a"],
        "mlp.c_proj.lora_b": params[prefix + "mlp.c_proj.lora_b"],
        "mlp.c_proj.alpha": params[prefix + "mlp.c_proj.alpha"],
    }


def gpt_forward(input_ids, attn_mask, params, n_head, n_layer):
    _, seq_len = input_ids.shape
    tok = params["wte.weight"][input_ids]
    pos = params["wpe.weight"][mx.arange(seq_len)]
    x = tok + pos.reshape(1, seq_len, -1)

    for layer in range(n_layer):
        x = transformer_block(x, attn_mask, block_view(params, layer), n_head)

    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])
    return x @ params["lm_head.weight"]


class GPT2Tokenizer:
    def __init__(self):
        import tiktoken

        self.encoder = tiktoken.get_encoding("gpt2")
        self.pad_token_id = self.encoder.eot_token
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, ids):
        return self.encoder.decode(ids)


def load_examples(path):
    examples = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["prompt"].strip()
            completion = row["completion"].strip()
            prefix = f"Instruction:\n{prompt}\n\nResponse:\n"
            full = prefix + completion
            examples.append((prefix, full))
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def build_dataset(path, tokenizer, block_size):
    examples = load_examples(path)
    dataset = []

    for prefix, full in examples:
        full_ids = tokenizer.encode(full)
        prefix_ids = tokenizer.encode(prefix)
        if len(full_ids) < 2:
            continue

        input_ids = full_ids[:-1][:block_size]
        labels = full_ids[1:][:block_size]
        prompt_target_cutoff = max(len(prefix_ids) - 1, 0)
        for i in range(min(prompt_target_cutoff, len(labels))):
            labels[i] = -100

        dataset.append(
            {
                "input_ids": np.array(input_ids, dtype=np.int64),
                "labels": np.array(labels, dtype=np.int64),
            }
        )

    if not dataset:
        raise ValueError("dataset became empty after preprocessing")
    return dataset


def make_batch(dataset, batch_size, block_size, pad_token_id):
    indices = np.random.randint(0, len(dataset), size=batch_size)
    input_ids = np.full((batch_size, block_size), pad_token_id, dtype=np.int64)
    labels = np.full((batch_size, block_size), -100, dtype=np.int64)
    attn_mask = np.zeros((batch_size, block_size), dtype=np.float32)

    for row, idx in enumerate(indices):
        item = dataset[idx]
        length = min(len(item["input_ids"]), block_size)
        input_ids[row, :length] = item["input_ids"][:length]
        labels[row, :length] = item["labels"][:length]
        attn_mask[row, :length] = 1.0

    return mx.array(input_ids), mx.array(labels), mx.array(attn_mask)


def top_k_sample(logits, top_k=40, temperature=0.8):
    logits_np = np.array(logits)
    logits_np = logits_np / max(temperature, 1e-4)
    top_k = min(top_k, logits_np.shape[-1])
    top_idx = np.argpartition(logits_np, -top_k)[-top_k:]
    top_logits = logits_np[top_idx]
    top_logits = top_logits - np.max(top_logits)
    probs = np.exp(top_logits)
    probs = probs / probs.sum()
    return int(np.random.choice(top_idx, p=probs))


def sample_text(prompt, tokenizer, params, n_head, n_layer, block_size, max_new_tokens=80, temperature=0.8, top_k=40, stream=False):
    ids = tokenizer.encode(prompt)
    if not ids:
        return ""

    if stream:
        print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        window = ids[-block_size:]
        input_ids = mx.array(np.array([window], dtype=np.int64))
        attn_mask = mx.ones((1, len(window)), dtype=mx.float32)

        logits = gpt_forward(input_ids, attn_mask, params, n_head, n_layer)
        next_id = top_k_sample(logits[0, -1], top_k=top_k, temperature=temperature)
        ids.append(next_id)
        if stream:
            print(tokenizer.decode([next_id]), end="", flush=True)

    if stream:
        print()

    return tokenizer.decode(ids)


def resolve_prompt(args):
    if args.sample_prompt is not None:
        return args.sample_prompt
    return input("Prompt: ")


def save_lora(params, config, path):
    payload = {
        "alpha": np.array([config["lora_alpha"]], dtype=np.float32),
        "rank": np.array([config["lora_rank"]], dtype=np.int32),
    }
    for name, value in params.items():
        if ".lora_" in name:
            payload[name] = np.array(value)
    np.savez(path, **payload)


def load_lora(params, path):
    payload = np.load(path)
    loaded = 0
    for name in payload.files:
        if name in {"alpha", "rank"}:
            continue
        if name not in params:
            raise KeyError(f"LoRA parameter {name} was not found in the current model")
        if tuple(params[name].shape) != payload[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint has {payload[name].shape}, model expects {tuple(params[name].shape)}"
            )
        params[name] = mx.array(payload[name].astype(np.float32))
        loaded += 1
    return loaded


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def heartbeat(step, total_steps, start_time, tokens_seen, phase=""):
    elapsed = time.time() - start_time
    avg = elapsed / max(step, 1)
    remaining = avg * max(total_steps - step, 0)
    tokens_per_sec = tokens_seen / max(elapsed, 1e-8)
    phase_text = f" | {phase}" if phase else ""
    print(
        f"step {step}/{total_steps} | elapsed {format_duration(elapsed)} ({elapsed:.1f}s) | eta {format_duration(remaining)} | {tokens_per_sec:.1f} tok/s{phase_text}",
        flush=True,
    )


def merge_params(params, trainable_params):
    merged = dict(params)
    merged.update(trainable_params)
    return merged


def extract_trainable_params(params):
    return {name: value for name, value in params.items() if ".lora_" in name}


def loss_fn(trainable_params, input_ids, labels, attn_mask, params, n_head, n_layer):
    merged_params = merge_params(params, trainable_params)
    logits = gpt_forward(input_ids, attn_mask, merged_params, n_head, n_layer)
    return cross_entropy(logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Load GPT-2 safetensors weights and train LoRA adapters in MLX.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--data", type=str, default="train.jsonl", help="JSONL file with prompt/completion fields")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face repo ID or local directory containing config.json and safetensors weights",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=None, help="Training/inference context length override")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-lora", type=str, default="mlx_gpt2_lora.npz")
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    np.random.seed(args.seed)

    params, lora_param_names, hf_config = init_model_from_hf(args.model, rank=args.rank, alpha=args.alpha)
    hf_config["lora_rank"] = args.rank
    hf_config["lora_alpha"] = args.alpha

    tokenizer = GPT2Tokenizer()
    model_block_size = hf_config["n_ctx"]
    if args.block_size is None:
        block_size = model_block_size
    else:
        if args.block_size <= 0:
            raise ValueError("--block-size must be positive")
        block_size = min(args.block_size, model_block_size)

    total_params = sum(
        int(np.prod(value.shape))
        for name, value in params.items()
        if hasattr(value, "shape") and ".alpha" not in name
    )
    trainable_params = sum(int(np.prod(params[name].shape)) for name in lora_param_names)

    print(f"loaded base model {args.model}")
    print(f"context length: {block_size} (model max {model_block_size})")
    print(f"trainable params: {trainable_params} / {total_params}")

    if args.load_lora is not None:
        loaded = load_lora(params, args.load_lora)
        print(f"loaded {loaded} LoRA tensors from {args.load_lora}")

    if args.mode == "train":
        dataset = build_dataset(args.data, tokenizer, block_size)
        if args.weight_decay:
            optimizer = optim.AdamW(
                learning_rate=args.lr,
                betas=[0.9, 0.999],
                eps=1e-8,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                learning_rate=args.lr,
                betas=[0.9, 0.999],
                eps=1e-8,
            )
        trainable_params_tree = extract_trainable_params(params)
        loss_and_grad = mx.value_and_grad(loss_fn)
        train_start = time.time()
        tokens_seen = 0

        print(f"loaded {len(dataset)} training examples")

        for step in range(1, args.steps + 1):
            heartbeat(step, args.steps, train_start, tokens_seen, phase="batch")
            input_ids, labels, attn_mask = make_batch(
                dataset,
                batch_size=args.batch_size,
                block_size=block_size,
                pad_token_id=tokenizer.pad_token_id,
            )
            tokens_seen += int(np.array(mx.sum(labels != -100)))
            heartbeat(step, args.steps, train_start, tokens_seen, phase="forward")
            loss, grads = loss_and_grad(
                trainable_params_tree,
                input_ids,
                labels,
                attn_mask,
                params,
                hf_config["n_head"],
                hf_config["n_layer"],
            )
            heartbeat(step, args.steps, train_start, tokens_seen, phase="update")
            optimizer.update(trainable_params_tree, grads)
            params.update(trainable_params_tree)
            mx.eval(trainable_params_tree, optimizer.state, loss)

            print(f"step {step:04d} | loss {loss.item():.4f}")

        save_lora(params, hf_config, args.save_lora)
        print(f"\nsaved LoRA adapters to {args.save_lora}")

    elif args.load_lora is None:
        print("running inference with freshly initialized LoRA adapters")

    prompt = resolve_prompt(args)
    print("\nsample:")
    text = sample_text(
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
