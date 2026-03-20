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

## gpt_peft_rlhf_mlx.py
Toy RLHF example built on top of `gpt_peft_mlx.py`. It supports interactive preference collection, a tiny reward model trained on chosen/rejected pairs, and a PPO-lite LoRA update.
```
pip install mlx safetensors huggingface_hub tiktoken
```

Create a prompts file:
```text
Write a tagline for a coffee shop.
Explain recursion simply.
Give a short pep talk for studying.
```

Collect preferences interactively:
```
python gpt_peft_rlhf_mlx.py --mode collect --model gpt2 --prompts-file prompts.txt --preference-data preferences.jsonl
```

Preference rows look like:
```json
{"prompt":"Write a tagline for a coffee shop.","chosen":"Warm cups, bright mornings.","rejected":"Coffee store words."}
```

Generate a toy starter preference file automatically:
```
python make_toy_preferences.py --prompts prompts.txt --output preferences.generated.jsonl --overwrite
```

Train the reward model and RL policy:
```
python gpt_peft_rlhf_mlx.py --mode train --model gpt2 --preference-data preferences.jsonl --reward-steps 100 --rl-steps 50 --save-lora mlx_gpt2_rlhf_lora.npz
```

Run inference with the RLHF-tuned LoRA:
```
python gpt_peft_rlhf_mlx.py --mode infer --model gpt2 --load-lora mlx_gpt2_rlhf_lora.npz
```

## gpt_peft_dpo_mlx.py
Toy DPO example built on top of `gpt_peft_mlx.py`. Unlike the RLHF example above, this one does not train a separate reward model. It directly optimizes chosen responses over rejected ones relative to a frozen reference policy.
```
pip install mlx safetensors huggingface_hub tiktoken
```

Train DPO with the starter preference pairs:
```
python gpt_peft_dpo_mlx.py --mode train --model gpt2 --preference-data preferences.jsonl --steps 100 --save-lora mlx_gpt2_dpo_lora.npz
```

Run inference with the DPO-tuned LoRA:
```
python gpt_peft_dpo_mlx.py --mode infer --model gpt2 --load-lora mlx_gpt2_dpo_lora.npz
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

## mlx/cifar10_vit_mlx.py
Vision Transformer image classification on CIFAR-10 in MLX.
```
pip install mlx matplotlib numpy
```

Train from scratch:
```
python mlx/cifar10_vit_mlx.py --mode train --epochs 20 --no-show
```

Resume training from saved weights:
```
python mlx/cifar10_vit_mlx.py --mode train --resume-from cifar10_vit_mlx.safetensors --epochs 10 --no-show
```

Evaluate on the CIFAR-10 test split:
```
python mlx/cifar10_vit_mlx.py --mode eval --weights-path cifar10_vit_mlx.safetensors
```

Render a small grid of test predictions:
```
python mlx/cifar10_vit_mlx.py --mode predict --weights-path cifar10_vit_mlx.safetensors --num-samples 16
```

## mlx/cifar10_gan_mlx.py
Class-conditional CIFAR-10 GAN training and sampling in MLX.
```
pip install mlx matplotlib numpy
```

Train from scratch:
```
python mlx/cifar10_gan_mlx.py --mode train --epochs 50 --no-show
```

Resume training from saved weights:
```
python mlx/cifar10_gan_mlx.py --mode train --resume-generator cifar10_gan_mlx_generator.safetensors --resume-discriminator cifar10_gan_mlx_discriminator.safetensors --epochs 20 --no-show
```

Generate class-conditioned samples:
```
python mlx/cifar10_gan_mlx.py --mode sample --generator-path cifar10_gan_mlx_generator.safetensors --prompts airplane cat dog ship
```

Generate unconditional samples:
```
python mlx/cifar10_gan_mlx.py --mode sample --generator-path cifar10_gan_mlx_generator.safetensors --unconditional --num-samples 64
```

## mlx/video_vit_mlx.py
Small video Vision Transformer classification in MLX using UCF101 clips and `ffmpeg` frame decoding.
```
pip install mlx matplotlib numpy
```

Prepare an extracted UCF101 directory like:
```text
/path/to/UCF-101/
  Basketball/
  Biking/
  HorseRiding/
  ...
```

Optional official split files can be downloaded from the UCF101 site and passed with `--train-split-file` and `--test-split-file`.

Train on a small subset of real video classes:
```
python mlx/video_vit_mlx.py --mode train --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 80 --max-test-videos-per-class 20 --no-show
```

Train without official split files using a simple local split:
```
python mlx/video_vit_mlx.py --mode train --videos-dir /path/to/UCF-101 --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 80 --max-test-videos-per-class 20 --no-show
```

Evaluate a saved checkpoint:
```
python mlx/video_vit_mlx.py --mode eval --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_vit_ucf101_mlx.safetensors
```

Render a few test-set predictions:
```
python mlx/video_vit_mlx.py --mode predict --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_vit_ucf101_mlx.safetensors --num-samples 4
```

## mlx/video_diffusion_mlx.py
Small class-conditional video diffusion in MLX using UCF101 clips, `ffmpeg` decoding, and a compact spatiotemporal U-Net.
```
pip install mlx matplotlib numpy
```

This example expects an extracted UCF101 directory, optionally paired with the official split files.

Train on a small subset of classes:
```
python mlx/video_diffusion_mlx.py --mode train --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 40 --max-test-videos-per-class 8 --no-show
```

Train without official split files using a simple local split:
```
python mlx/video_diffusion_mlx.py --mode train --videos-dir /path/to/UCF-101 --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 40 --max-test-videos-per-class 8 --no-show
```

Render the forward diffusion process for one decoded training clip:
```
python mlx/video_diffusion_mlx.py --mode forward --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_diffusion_ucf101_mlx.safetensors --no-show
```

Generate class-conditioned video samples:
```
python mlx/video_diffusion_mlx.py --mode sample --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_diffusion_ucf101_mlx.safetensors --prompts Basketball Surfing Typing --no-show
```

Generate unconditional video samples:
```
python mlx/video_diffusion_mlx.py --mode sample --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_diffusion_ucf101_mlx.safetensors --unconditional --num-samples 4 --no-show
```

## mlx/image_to_video_diffusion_mlx.py
Small image-to-video diffusion in MLX. It trains on real UCF101 clips, uses the first frame as the conditioning image, and generates the rest of the clip from that reference.
```
pip install mlx matplotlib numpy
```

Train on a small subset of classes:
```
python mlx/image_to_video_diffusion_mlx.py --mode train --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 40 --max-test-videos-per-class 8 --no-show
```

Render the forward diffusion process for a training clip:
```
python mlx/image_to_video_diffusion_mlx.py --mode forward --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path image_to_video_diffusion_ucf101_mlx.safetensors --no-show
```

Generate a video from a standalone input image:
```
python mlx/image_to_video_diffusion_mlx.py --mode sample --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path image_to_video_diffusion_ucf101_mlx.safetensors --input-image /path/to/reference.png --no-show
```

If `--input-image` is omitted, the script falls back to the first frame of one decoded training video as the reference image.

## mlx/video_vae_mlx.py
Small spatiotemporal video VAE in MLX. This is a useful next stage before latent video diffusion because it learns a compressed clip representation and reconstructs real UCF101 videos from that latent space.
```
pip install mlx matplotlib numpy
```

Train the VAE on a small subset of classes:
```
python mlx/video_vae_mlx.py --mode train --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 40 --max-test-videos-per-class 8 --no-show
```

Render reconstructions from the saved VAE:
```
python mlx/video_vae_mlx.py --mode reconstruct --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_vae_ucf101_mlx.safetensors --no-show
```

Sample random clips from the VAE prior:
```
python mlx/video_vae_mlx.py --mode sample --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path video_vae_ucf101_mlx.safetensors --num-samples 4 --no-show
```

## mlx/latent_video_diffusion_mlx.py
Latent video diffusion in MLX on top of the pretrained `mlx/video_vae_mlx.py` autoencoder. This is the natural next step after the VAE: diffuse in compressed clip latents and decode samples back to RGB video.
```
pip install mlx matplotlib numpy
```

Train the VAE first, then train latent diffusion with the saved VAE checkpoint:
```
python mlx/latent_video_diffusion_mlx.py --mode train --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --vae-weights-path video_vae_ucf101_mlx.safetensors --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 40 --max-test-videos-per-class 8 --no-show
```

Generate class-conditioned latent video samples:
```
python mlx/latent_video_diffusion_mlx.py --mode sample --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --vae-weights-path video_vae_ucf101_mlx.safetensors --weights-path latent_video_diffusion_ucf101_mlx.safetensors --prompts Basketball Surfing Typing --no-show
```

Generate unconditional latent video samples:
```
python mlx/latent_video_diffusion_mlx.py --mode sample --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --vae-weights-path video_vae_ucf101_mlx.safetensors --weights-path latent_video_diffusion_ucf101_mlx.safetensors --unconditional --num-samples 4 --no-show
```

## mlx/frame_interpolation_mlx.py
Small frame interpolation model in MLX. It learns to predict the middle frame of a short real-video triplet from the left and right frames.
```
pip install mlx matplotlib numpy
```

Train on a small subset of UCF101 classes:
```
python mlx/frame_interpolation_mlx.py --mode train --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --classes Basketball,Biking,HorseRiding,Surfing,Typing --max-train-videos-per-class 80 --max-test-videos-per-class 20 --no-show
```

Render a few left/target/predicted/right interpolation examples from the saved checkpoint:
```
python mlx/frame_interpolation_mlx.py --mode predict --videos-dir /path/to/UCF-101 --train-split-file /path/to/ucfTrainTestlist/trainlist01.txt --test-split-file /path/to/ucfTrainTestlist/testlist01.txt --weights-path frame_interpolation_ucf101_mlx.safetensors --no-show
```
