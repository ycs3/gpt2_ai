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

## gpt_peft_mlx_quant4.py
4-bit MLX quantization keeps the frozen GPT-2 base weights in grouped affine INT4 form and leaves LoRA weights trainable in float32.
```
python gpt_peft_mlx_quant4.py --mode train --model gpt2 --data train.jsonl --save-lora mlx_gpt2_lora_quant4.npz
python gpt_peft_mlx_quant4.py --mode infer --model gpt2 --load-lora mlx_gpt2_lora_quant4.npz
```

## gpt_peft_np_quant4.py
4-bit NumPy quantization packs two affine INT4 weights per byte for the frozen base model and dequantizes on demand during forward passes.
```
python gpt_peft_np_quant4.py --mode train --model gpt2 --data train.jsonl --save-lora numpy_gpt2_lora_quant4.npz
python gpt_peft_np_quant4.py --mode infer --model gpt2 --load-lora numpy_gpt2_lora_quant4.npz
```

## cifar10_diffusion_mlx.py
Text-conditional CIFAR-10 diffusion training and sampling in MLX.
```
pip install mlx matplotlib numpy
```

Train from scratch:
```
python cifar10_diffusion_mlx.py --mode train --sample-steps 100 --no-show
```

Resume training from saved weights:
```
python cifar10_diffusion_mlx.py --mode train --resume-from diffusion_cifar10_labels_mlx.safetensors --sample-steps 100 --no-show
```

Render the forward diffusion process:
```
python cifar10_diffusion_mlx.py --mode forward --weights-path diffusion_cifar10_labels_mlx.safetensors
```

Generate class-conditioned samples:
```
python cifar10_diffusion_mlx.py --mode sample --weights-path diffusion_cifar10_labels_mlx.safetensors --prompts airplane cat dog ship --sample-steps 50
```

Generate unconditional samples:
```
python cifar10_diffusion_mlx.py --mode sample --weights-path diffusion_cifar10_labels_mlx.safetensors --unconditional --num-samples 64 --sample-steps 50
```
