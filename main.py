import argparse, os, sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import gc

from models.lenet import LeNet
from models.resnet import ResNet18
from loader.loaders import get_dataloader
from training.train import Trainer
from training.EASGD_optimizer import EA_SGD
from training.EntropySGD import EntropySGD
from utils.logger import Logger
from torch.optim import SGD, Adam

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_model(model_name, num_classes=10):
    """Get model based on configuration"""
    if model_name == "LeNet":
        return LeNet(num_classes=num_classes)
    elif model_name == "ResNet18":
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def get_optimizer(optimizer_config, model):
    """Get optimizer based on configuration"""
    opt_name = optimizer_config['name']
    
    if opt_name == "EASGD":
        # Get k_sparsity_schedule if it exists, otherwise use None
        k_sparsity_schedule = optimizer_config.get('k_sparsity_schedule', None)
        
        return EA_SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            inner_lr=optimizer_config['inner_lr'],
            inner_steps=optimizer_config['inner_steps'],
            thermal_noise=optimizer_config.get('thermal_noise', 1e-3),
            projection_freq=optimizer_config['projection_freq'],
            k_sparsity=optimizer_config.get('k_sparsity', 0.1),  # Default value if not provided
            gamma=optimizer_config['gamma'],
            momentum=optimizer_config.get('momentum', 0),
            weight_decay=optimizer_config['weight_decay'],
            kfac_update_freq=optimizer_config['kfac_update_freq'],
            k_sparsity_schedule=k_sparsity_schedule  # Pass the schedule
        )
    elif opt_name == "SGD":
        return SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0),
            weight_decay=optimizer_config['weight_decay']
        )
    elif opt_name == "Adam":
        return Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif opt_name == "EntropySGD":
        return EntropySGD(
            model.parameters(),
            config={
                'lr': optimizer_config['lr'],
                'momentum': optimizer_config.get('momentum', 0),
                'weight_decay': optimizer_config['weight_decay'],
                'nesterov': optimizer_config.get('nesterov', True),
                'L': optimizer_config['L'],
                'eps': optimizer_config['eps'],
                'g0': optimizer_config['g0'],
                'g1': optimizer_config['g1']
            }
        )

def main():
    print("Starting main.py")
    parser = argparse.ArgumentParser(description='EAD-SGD Training')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    print("Setting up...")
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader = get_dataloader(
        config['dataset'], config['data_path'], 
        config['batch_size'], config['num_workers']
    )

    # Dataset configuration
    if config['dataset'] == 'CIFAR10':
        num_classes = 10
    elif config['dataset'] == 'MNIST':
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")
    
    # Create model
    model = get_model(config['model'], num_classes).to(device)

    # Create logger with num_classes parameter
    logger = Logger(config['logging']['log_dir'], 
                use_tensorboard=config['logging']['use_tensorboard'],
                num_classes=num_classes)
    
    # Create optimizer
    optimizer = get_optimizer(config['optimizer'], model)
    
    # Set model reference in optimizer if it's EA-SGD
    if config['optimizer']['name'] == "EASGD":
        optimizer.set_model(model)
    
    # Create logger directory if it doesn't exist
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = Logger(log_dir, 
                   use_tensorboard=config['logging']['use_tensorboard'],
                   num_classes=num_classes,
                   optimizer=optimizer)
    
    # Save the config to the log directory for reference
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print("Setup complete!")
    
    # Handle resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state if it exists
        if trainer.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Resumed from checkpoint: {args.resume}")
    
    try:
        print("Beginning training...")
        # Create trainer and start training
        trainer = Trainer(model, optimizer, train_loader, test_loader, 
                         device, logger, config)
        trainer.train()
        print("Training Complete!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up DataLoader workers
        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'close'):
            train_loader.dataset.close()
        if hasattr(test_loader, 'dataset') and hasattr(test_loader.dataset, 'close'):
            test_loader.dataset.close()
            
        # Force garbage collection
        gc.collect()
        
        # If using CUDA, clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.close()
        print("Cleanup completed, exiting...")

    sys.exit(0)

    # # Create trainer and start training
    # trainer = Trainer(model, optimizer, train_loader, test_loader, 
    #                  device, logger, config)
    # trainer.train()
    
if __name__ == '__main__':
    main()