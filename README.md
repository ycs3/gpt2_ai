# gpt2_ai
GPT-2 AI generated code for learning

## gpt_peft_mlx.py
Upgrading CUDA on Linux
```
apt-get update
apt-get install -y cuda-toolkit-12-9
ln -sfn /usr/local/cuda-12.9 /usr/local/cuda
pip install "mlx[cuda12]" safetensors huggingface_hub tiktoken
```