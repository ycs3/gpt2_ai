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
Running training/inference
```
python gpt_peft_mlx.py --mode train --model gpt2 --data train.jsonl --save-lora mlx_gpt2_lora.npz
python gpt_peft_mlx.py --mode infer --model gpt2 --load-lora mlx_gpt2_lora.npz
python gpt_peft_mlx.py --mode infer --model gpt2 # run gpt2 as-is
```