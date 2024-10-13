# image-classification-with-lightning

PyTorch Lightning + Hydra + WandB

Simple image classification for toy datasets (MNIST, CIFAR-10)

## Setup

```bash
poetry install --no-root
poetry run wandb login
```

## Usage

```bash
# train
poetry run python src/train.py experiment={mnist_simple|mnist_conv|cifar10_conv}

# test
poetry run python src/test.py --ckpt <path to checkpoint>
```
