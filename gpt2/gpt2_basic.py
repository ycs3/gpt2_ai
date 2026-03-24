import tiktoken
from safetensors.numpy import load_file
import numpy as np

def softmax(x):
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    return (e_x / np.sum(e_x, axis=1, keepdims=True))

def layer_norm(x, scale, shift):
    x = ((x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True))
    x = (x*scale)+shift
    return x

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gpt2():
    encoder = tiktoken.get_encoding("gpt2")
    state = load_file("model.safetensors")

    tokens = [40, 588, 616, 649]

    for t in range(20):
        x = np.array(tokens, dtype=np.int64)
        x = state["wte.weight"][x]

        t = np.array([i for i in range(len(tokens))], dtype=np.int64)
        t = state["wpe.weight"][t]

        x = x + t

        for layer in range(12):
            p = f"h.{layer}."
            x_orig = np.array(x)
            x = layer_norm(x, state[p+"ln_1.weight"], state[p+"ln_1.bias"])

            x = x @ state[p+"attn.c_attn.weight"] + state[p+"attn.c_attn.bias"]

            q, k, v = x[:,:768], x[:,768:768*2], x[:,768*2:]

            n_head = 12
            head_sz = 768 // n_head
            x = []
            for i in range(n_head):
                q_ = q[:,head_sz*i:head_sz*(i+1)]
                k_ = k[:,head_sz*i:head_sz*(i+1)]
                v_ = v[:,head_sz*i:head_sz*(i+1)]

                x_ = q_ @ k_.T
                sz = x_.shape[0]
                x_ = x_ + (np.triu([-np.inf for _ in range(sz)], 1))
                x_ = x_ / np.sqrt(head_sz)
                x_ = softmax(x_)
                x_ = x_ @ v_
                x.append(x_)

            x = np.concatenate(x, axis=1)

            x = x @ state[p+"attn.c_proj.weight"] + state[p+"attn.c_proj.bias"]
            x = x_orig + x

            x_orig = np.array(x)

            x = layer_norm(x, state[p+"ln_2.weight"], state[p+"ln_2.bias"])

            x = x @ state[p+"mlp.c_fc.weight"] + state[p+"mlp.c_fc.bias"]
            x = gelu(x)
            x = x @ state[p+"mlp.c_proj.weight"] + state[p+"mlp.c_proj.bias"]

            x = x_orig + x

        x = layer_norm(x, state["ln_f.weight"], state["ln_f.bias"])
        x = x @ state["wte.weight"].T
        next_token = np.argmax(x[-1])

        tokens.append(next_token)        
        print(encoder.decode(tokens))

if __name__ == '__main__':
    gpt2()
