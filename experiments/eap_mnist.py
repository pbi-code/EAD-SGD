import torch
import torch.nn as nn
import sys
import os
import json

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.lenet import LeNet
from training.EASGD_optimizer import EA_SGD
from data.loaders import get_dataloader
from utils.logger import Logger
from utils.metrics import Accuracy, AverageMeter, GeneralizationAnalyzer, SharpnessCalculator

# Configuration - we'll create a simple config since we're not using YAML files
config = {
    'dataset': 'MNIST',
    'model': 'LeNet',
    'training': {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 0
    },
    'ea_sgd': {
        'inner_lr': 0.1,
        'inner_steps': 20,
        'projection_freq': 5,
        'k_sparsity': 0.1,
        'gamma': 0.03,
        'kfac_update_freq': 100
    }
}

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, test_loader = get_dataloader(
        config['dataset'], 
        './data', 
        config['training']['batch_size'], 
        num_workers=4
    )
    
    # Initialize model
    model = LeNet().to(device)
    
    # Initialize optimizer
    optimizer = EA_SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        inner_lr=config['ea_sgd']['inner_lr'],
        inner_steps=config['ea_sgd']['inner_steps'],
        projection_freq=config['ea_sgd']['projection_freq'],
        k_sparsity=config['ea_sgd']['k_sparsity'],
        gamma=config['ea_sgd']['gamma'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay'],
        kfac_update_freq=config['ea_sgd']['kfac_update_freq']
    )
    
    # Set model reference in optimizer
    optimizer.set_model(model)
    
    # Initialize logger
    log_dir = f"./logs/mnist_ea_sgd_{config['ea_sgd']['gamma']}_{config['ea_sgd']['k_sparsity']}"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir, use_tensorboard=True, num_classes=10)
    
    # Initialize metrics
    criterion = nn.CrossEntropyLoss()
    gen_analyzer = GeneralizationAnalyzer(10)
    sharpness_calculator = SharpnessCalculator(model)
    
    # Training loop
    best_acc = 0.0
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Training phase
        model.train()
        loss_meter = AverageMeter()
        acc_meter = Accuracy()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Get activations and gradients for K-FAC
            activations = model.activations
            gradients = model.gradients
            
            # Optimization step
            optimizer.step(
                data_loader=train_loader,
                activations=activations,
                gradients=gradients
            )
            
            # Update metrics
            loss_meter.update(loss.item(), data.size(0))
            acc = acc_meter.update(output, target)
            
            # Update generalization analyzer with batch data
            gen_analyzer.update_batch(output, target)
        
        train_loss = loss_meter.avg
        train_acc = acc_meter.avg
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Testing phase
        model.eval()
        test_loss, test_acc = test_model(model, test_loader, criterion, device, gen_analyzer)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Update generalization analyzer with epoch data
        gen_analyzer.update_epoch(epoch, train_loss, train_acc, test_loss, test_acc)
        
        # Log metrics
        logger.log_metrics(epoch, train_loss, train_acc, test_loss, test_acc, 0)  # Time not tracked for simplicity
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, f'{log_dir}/best_model.pth')
        
        # Print epoch summary
        print(f'Epoch {epoch:3d}/{config["training"]["epochs"]}: '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        
        # Calculate and log sharpness periodically
        if epoch % 10 == 0:
            sharpness = sharpness_calculator.calculate_sharpness(criterion, test_loader, device)
            print(f"Sharpness of minimum: {sharpness:.6f}")
            if logger.use_tensorboard:
                logger.writer.add_scalar('Generalization/Sharpness', sharpness, epoch)
    
    # Generate comprehensive generalization report
    gen_report = gen_analyzer.generate_report(
        save_path=os.path.join(log_dir, 'generalization_report.json')
    )
    
    # Plot generalization curves
    plot_dir = os.path.join(log_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    gen_analyzer.plot_loss_curves(os.path.join(plot_dir, 'loss_curves.png'))
    gen_analyzer.plot_accuracy_curves(os.path.join(plot_dir, 'accuracy_curves.png'))
    gen_analyzer.plot_calibration_curve(os.path.join(plot_dir, 'calibration_curve.png'))
    
    # Add sharpness to the report
    final_sharpness = sharpness_calculator.calculate_sharpness(criterion, test_loader, device)
    gen_report['final_sharpness'] = final_sharpness
    with open(os.path.join(log_dir, 'generalization_report.json'), 'w') as f:
        json.dump(gen_report, f, indent=4)
    
    print(f'Final sharpness: {final_sharpness:.6f}')
    print(f'Training completed. Best accuracy: {best_acc:.2f}%')
    
    # Close logger
    logger.close()
    
    return train_losses, test_losses, train_accs, test_accs

def test_model(model, test_loader, criterion, device, gen_analyzer):
    """Evaluate the model on the test set"""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = Accuracy()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            loss_meter.update(loss.item(), data.size(0))
            acc_meter.update(output, target)
            
            # Update generalization analyzer with test data
            gen_analyzer.update_batch(output, target)
    
    return loss_meter.avg, acc_meter.avg

if __name__ == "__main__":
    train_model()