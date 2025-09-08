# EAD-SGD: Entropy Anisotropy Dual Stochastic Gradient Descent

This repository implements the EAD-SGD optimizer, which combines:
- Local entropy exploration via SGLD (M-projection)
- Information-rich subspace selection via SNR-based Walsh-Hadamard Transform (WHT) compression (E-projection)
- K-FAC approximation for Fisher Information Matrix preconditioning

## Key Features

1. EA-SGD Optimizer: Implements the Entropy Anisotropy Dual SGD algorithm

2. K-FAC Approximation: Efficient Fisher Information Matrix approximation

3. WHT Compression: Walsh-Hadamard Transform for gradient compression

4. Local Entropy Exploration: SGLD sampling for wide minima discovery (see Entropy-SGD approach)

5. Comprehensive Logging: TensorBoard integration and JSON logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PBJacket/EAD-SGD.git
cd EAD-SGD
```

2. Install requirements:
```python -m venv venv```
```source venv/bin/activate``` On Windows: venv\Scripts\activate
```pip install -r requirements.txt```

## Usage

1. Training on CIFAR-10:
```python main.py --config configs/cifar10.yaml```

2. Training on MNIST:
```python main.py --config configs/mnist.yaml```

3. Resuming from a checkpoint:
```python main.py --config configs/cifar10.yaml --resume logs/cifar10/best_model.pth``` 

## Results

The optimizer is designed to find wider minima in the loss landscape, which typically leads to better generalization performance. Results on CIFAR-10 and MNIST will be documented here.

## Citation
If you use this code in your research, please cite:

@software{ead_sgd,
  title = {EAD-SGD: Entropy Anisotropy Dual Stochastic Gradient Descent},
  author = {Prahlad Iyengar},
  year = {2025},
  url = {https://github.com/PBJacket/EAD-SGD}
}

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## File Structure

EAD-SGD/
├── configs/               # Experiment configurations
│   ├── base.yaml         # Default parameters
│   ├── cifar10.yaml      # CIFAR-10 specific
│   └── mnist.yaml        # MNIST specific
│
├── data/                 # Data handling
│   ├── __init__.py
│   └── loaders.py        # Dataset loaders
│
├── models/               # Model architectures
│   ├── __init__.py
│   ├── resnet.py         # ResNet model for CIFAR-10
│   └── lenet.py          # LeNet for MNIST
│
├── training/             # Training infrastructure
│   ├── __init__.py
│   ├── train.py           # Main training loop
│   ├── losses.py          # Custom loss functions
│   └── EASGD_optimizer.py # EA-SGD optimizer
│
├── visualization/        # Analysis tools
│   ├── __init__.py
│   └── augmentation.py   # Visualization of augmentations
│
├── utils/                # Helper functions
│   ├── __init__.py
│   ├── logger.py         # Logging utilities
│   └── metrics.py        # Evaluation metrics
│
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # This guide