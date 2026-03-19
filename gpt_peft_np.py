import argparse
import json
import os
import time

import numpy as np


def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add",
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-ensure_tensor(other))

    def __rsub__(self, other):
        return ensure_tensor(other) - self

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul",
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = ensure_tensor(other)
        return self * other.pow(-1.0)

    def pow(self, exponent):
        out = Tensor(self.data ** exponent, requires_grad=self.requires_grad, _children=(self,), _op="pow")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * exponent * (self.data ** (exponent - 1.0))

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = ensure_tensor(other)
        out = Tensor(
            np.matmul(self.data, other.data),
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul",
        )

        def _backward():
            if self.requires_grad:
                grad_self = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
                self.grad += unbroadcast(grad_self, self.data.shape)
            if other.requires_grad:
                grad_other = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
                other.grad += unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, _children=(self,), _op="transpose")
        inverse = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.transpose(*inverse)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], requires_grad=self.requires_grad, _children=(self,), _op="getitem")

        def _backward():
            if self.requires_grad:
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar tensors")
            grad = np.ones_like(self.data, dtype=np.float32)
        self.grad = grad.astype(np.float32)

        topo = []
        visited = set()

        def build(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        for node in reversed(topo):
            node._backward()


def ensure_tensor(value):
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


def softmax(x, axis=-1):
    shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / exps.sum(axis=axis, keepdims=True)
    out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")

    def _backward():
        if not x.requires_grad:
            return
        grad = out.grad
        dot = np.sum(grad * probs, axis=axis, keepdims=True)
        x.grad += probs * (grad - dot)

    out._backward = _backward
    return out


def gelu(x):
    coeff = np.float32(np.sqrt(2.0 / np.pi))
    inner = x.data + 0.044715 * (x.data ** 3)
    tanh_inner = np.tanh(coeff * inner)
    out_data = 0.5 * x.data * (1.0 + tanh_inner)
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="gelu")

    def _backward():
        if not x.requires_grad:
            return
        sech2 = 1.0 - tanh_inner ** 2
        inner_grad = coeff * (1.0 + 3.0 * 0.044715 * (x.data ** 2))
        local = 0.5 * (1.0 + tanh_inner) + 0.5 * x.data * sech2 * inner_grad
        x.grad += out.grad * local

    out._backward = _backward
    return out


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x.data - mean) * inv_std
    out_data = x_hat * gamma.data + beta.data
    out = Tensor(
        out_data,
        requires_grad=x.requires_grad or gamma.requires_grad or beta.requires_grad,
        _children=(x, gamma, beta),
        _op="layer_norm",
    )

    def _backward():
        grad_out = out.grad
        if gamma.requires_grad:
            axes = tuple(range(grad_out.ndim - 1))
            gamma.grad += np.sum(grad_out * x_hat, axis=axes)
        if beta.requires_grad:
            axes = tuple(range(grad_out.ndim - 1))
            beta.grad += np.sum(grad_out, axis=axes)
        if x.requires_grad:
            d_xhat = grad_out * gamma.data
            dvar = np.sum(d_xhat * (x.data - mean) * -0.5 * (var + eps) ** (-1.5), axis=-1, keepdims=True)
            dmean = np.sum(d_xhat * -inv_std, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * (x.data - mean), axis=-1, keepdims=True)
            dx = d_xhat * inv_std + dvar * 2.0 * (x.data - mean) / x.data.shape[-1] + dmean / x.data.shape[-1]
            x.grad += dx

    out._backward = _backward
    return out


def cross_entropy(logits, targets, ignore_index=-100):
    logits_data = logits.data
    flat_logits = logits_data.reshape(-1, logits_data.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    if not np.any(valid):
        raise ValueError("batch has no valid targets")

    valid_logits = flat_logits[valid]
    valid_targets = flat_targets[valid]
    shifted = valid_logits - np.max(valid_logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    loss_value = -np.mean(log_probs[np.arange(len(valid_targets)), valid_targets])
    out = Tensor(loss_value, requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")

    def _backward():
        if not logits.requires_grad:
            return
        probs = np.exp(log_probs)
        probs[np.arange(len(valid_targets)), valid_targets] -= 1.0
        probs /= len(valid_targets)
        grad = np.zeros_like(flat_logits, dtype=np.float32)
        grad[valid] = probs
        logits.grad += grad.reshape(logits_data.shape) * out.grad

    out._backward = _backward
    return out


def lora_linear(x, weight, bias, lora_a, lora_b, alpha):
    base = x @ weight + bias
    rank = lora_a.data.shape[1]
    scale = alpha / rank
    delta = (x @ lora_a) @ lora_b
    return base + delta * scale


def causal_self_attention(x, attn_mask, block, n_head):
    batch_size, seq_len, n_embd = x.data.shape
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
    causal = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e9
    scores = scores + causal.reshape(1, 1, seq_len, seq_len)

    if attn_mask is not None:
        key_mask = (1.0 - attn_mask.astype(np.float32)).reshape(batch_size, 1, 1, seq_len) * -1e9
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
        "lora_a": Tensor(np.random.randn(in_dim, rank).astype(np.float32) * 0.01, requires_grad=True),
        "lora_b": Tensor(np.zeros((rank, out_dim), dtype=np.float32), requires_grad=True),
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


def init_model_from_hf(model_name="gpt2", rank=4, alpha=8.0):
    import torch
    from huggingface_hub import snapshot_download

    model_dir = snapshot_download(
        repo_id=model_name,
        allow_patterns=["config.json", "*.bin", "*.safetensors", "*.index.json"],
    )

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required when the checkpoint is not stored as pytorch_model.bin") from exc

        safetensors_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(
                f"Could not find pytorch_model.bin or model.safetensors in {model_dir}. "
                "Sharded checkpoints are not supported in this teaching script."
            )
        state = load_file(safetensors_path)

    state = maybe_unwrap_state_dict(state)

    n_embd = config["n_embd"]
    n_head = config["n_head"]
    n_layer = config["n_layer"]
    block_size = config["n_positions"]

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

    params["wte.weight"] = Tensor(state[wte_key].float().numpy(), requires_grad=False)
    params["wpe.weight"] = Tensor(state[wpe_key].float().numpy(), requires_grad=False)
    params["ln_f.weight"] = Tensor(state[ln_f_weight_key].float().numpy(), requires_grad=False)
    params["ln_f.bias"] = Tensor(state[ln_f_bias_key].float().numpy(), requires_grad=False)
    params["lm_head.weight"] = Tensor(state[wte_key].float().numpy().T, requires_grad=False)

    for layer in range(n_layer):
        prefix = f"h.{layer}."
        for name in hf_weight_key_names(layer):
            hf_name = resolve_key_name(
                state,
                "transformer." + name,
                name,
                "model.transformer." + name,
            )
            params[name] = Tensor(state[hf_name].float().numpy(), requires_grad=False)

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
            params[module_name + ".alpha"] = alpha
            lora_params.extend([pair["lora_a"], pair["lora_b"]])

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
    pos = params["wpe.weight"][np.arange(seq_len)]
    x = tok + pos.reshape(1, seq_len, -1)

    for layer in range(n_layer):
        x = transformer_block(x, attn_mask, block_view(params, layer), n_head)

    x = layer_norm(x, params["ln_f.weight"], params["ln_f.bias"])
    return x @ params["lm_head.weight"]


class Adam:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay:
                grad = grad + self.weight_decay * p.data
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


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

    return input_ids, labels, attn_mask


def top_k_sample(logits, top_k=40, temperature=0.8):
    logits = logits / max(temperature, 1e-4)
    top_k = min(top_k, logits.shape[-1])
    top_idx = np.argpartition(logits, -top_k)[-top_k:]
    top_logits = logits[top_idx]
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
        input_ids = np.full((1, len(window)), tokenizer.pad_token_id, dtype=np.int64)
        attn_mask = np.ones((1, len(window)), dtype=np.float32)
        input_ids[0, :] = window

        logits = gpt_forward(input_ids, attn_mask, params, n_head, n_layer).data
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
            payload[name] = value.data
    np.savez(path, **payload)


def load_lora(params, path):
    payload = np.load(path)
    loaded = 0
    for name in payload.files:
        if name in {"alpha", "rank"}:
            continue
        if name not in params:
            raise KeyError(f"LoRA parameter {name} was not found in the current model")
        target = params[name]
        if not isinstance(target, Tensor):
            raise TypeError(f"Expected Tensor parameter for {name}")
        if target.data.shape != payload[name].shape:
            raise ValueError(
                f"Shape mismatch for {name}: checkpoint has {payload[name].shape}, model expects {target.data.shape}"
            )
        target.data[...] = payload[name].astype(np.float32)
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
    avg = elapsed / step
    remaining = avg * max(total_steps - step, 0)
    tokens_per_sec = tokens_seen / max(elapsed, 1e-8)
    phase_text = f" | {phase}" if phase else ""
    print(
        f"step {step}/{total_steps} | elapsed {format_duration(elapsed)} ({elapsed:.1f}s) | eta {format_duration(remaining)} | {tokens_per_sec:.1f} tok/s{phase_text}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Load HF GPT-2 weights and train LoRA adapters in NumPy.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--data", type=str, default="train.jsonl", help="JSONL file with prompt/completion fields")
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=None, help="Training/inference context length override")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-lora", type=str, default="numpy_gpt2_lora.npz")
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--sample-prompt", type=str, default=None)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    np.random.seed(args.seed)

    params, lora_params, hf_config = init_model_from_hf(args.model, rank=args.rank, alpha=args.alpha)
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

    total_params = sum(v.data.size for v in params.values() if isinstance(v, Tensor))
    trainable_params = sum(p.data.size for p in lora_params)

    print(f"loaded base model {args.model}")
    print(f"context length: {block_size} (model max {model_block_size})")
    print(f"trainable params: {trainable_params} / {total_params}")

    if args.load_lora is not None:
        loaded = load_lora(params, args.load_lora)
        print(f"loaded {loaded} LoRA tensors from {args.load_lora}")

    if args.mode == "train":
        dataset = build_dataset(args.data, tokenizer, block_size)
        optimizer = Adam(lora_params, lr=args.lr, weight_decay=args.weight_decay)
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
            tokens_seen += int(np.sum(labels != -100))
            heartbeat(step, args.steps, train_start, tokens_seen, phase="forward")
            optimizer.zero_grad()
            logits = gpt_forward(input_ids, attn_mask, params, hf_config["n_head"], hf_config["n_layer"])
            heartbeat(step, args.steps, train_start, tokens_seen, phase="backward")
            loss = cross_entropy(logits, labels)
            loss.backward()
            heartbeat(step, args.steps, train_start, tokens_seen, phase="update")
            optimizer.step()

            print(f"step {step:04d} | loss {loss.data.item():.4f}")

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
