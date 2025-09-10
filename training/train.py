import time
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os

from utils.metrics import Accuracy, AverageMeter, GeneralizationAnalyzer, SharpnessCalculator

class Trainer:
    """
    Main training class for EAD-SGD with enhanced generalization metrics
    """
    def __init__(self, model, optimizer, train_loader, test_loader, device, logger, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        self.config = config
        
        # Register hooks for KFAC
        self.model.register_hooks()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Initialize generalization analyzer
        num_classes = 10  # Adjust based on your dataset
        self.gen_analyzer = GeneralizationAnalyzer(num_classes)
        
        # Initialize sharpness calculator
        self.sharpness_calculator = SharpnessCalculator(self.model)
    
    def _get_scheduler(self):
        """Create learning rate scheduler based on configuration"""
        if 'scheduler' not in self.config['training']:
            return None
            
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config.get('type', None)
        
        if scheduler_type == "multistep":
            milestones = scheduler_config.get('milestones', [])
            gamma = scheduler_config.get('gamma', 0.1)
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=gamma
            )
        elif scheduler_type == "step":
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "cosine":
            T_max = scheduler_config.get('T_max', self.config['training']['epochs'])
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max
            )
        elif scheduler_type == "exponential":
            gamma = scheduler_config.get('gamma', 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Train for one epoch with generalization tracking"""
        self.model.train()
        loss_meter = AverageMeter()
        acc_meter = Accuracy()
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Check if optimizer is EA_SGD to pass extra arguments
            if hasattr(self.optimizer, 'is_ea_sgd') and self.optimizer.is_ea_sgd:
                # Get activations and gradients for K-FAC
                activations = self.model.activations
                gradients = self.model.gradients
                
                # Optimization step with EA-SGD specific arguments
                self.optimizer.step(
                    data_loader=self.train_loader,
                    activations=activations,
                    gradients=gradients
                )
            else:
                # Standard optimizer step without extra arguments
                self.optimizer.step()
            
            # Update metrics
            loss_meter.update(loss.item(), data.size(0))
            acc = acc_meter.update(output, target)
            
            # Update generalization analyzer with batch data
            self.gen_analyzer.update_batch(output, target)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}',
                'Acc': f'{acc:.2f}%'
            })
    
        return loss_meter.avg, acc_meter.avg
    
    def test(self):
        """Evaluate the model on the test set with generalization tracking"""
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = Accuracy()
        
        # Reset generalization analyzer for test set
        self.gen_analyzer.reset()
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss_meter.update(loss.item(), data.size(0))
                acc_meter.update(output, target)
                
                # Update generalization analyzer with test data
                self.gen_analyzer.update_batch(output, target)
        
        return loss_meter.avg, acc_meter.avg
    
    def calculate_sharpness(self):
        """Calculate sharpness of the current minimum"""
        print("Calculating sharpness at current minimum...")
        return self.sharpness_calculator.calculate_sharpness(
            self.criterion, self.test_loader, self.device
        )
    
    def train(self):
        """Main training loop with enhanced generalization tracking"""
        best_acc = 0.0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluate on test set
            test_loss, test_acc = self.test()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Update generalization analyzer with epoch data
            self.gen_analyzer.update_epoch(epoch, train_loss, train_acc, test_loss, test_acc)
            
            # Log metrics
            self.logger.log_metrics(epoch, train_loss, train_acc, test_loss, test_acc, epoch_time)
            
            # Save best model
            # print("Saving best model...")
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': self.config
                }, f'{self.logger.log_dir}/best_model.pth')
            # print("Best model saved!")
            
            # Print epoch summary
            print(f'Epoch {epoch:3d}/{self.config["training"]["epochs"]}: '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
                  f'Time: {epoch_time:.2f}s')
            
            # Calculate and log sharpness periodically
            if epoch % 10 == 0 and epoch != self.config['training']['epochs']:
                print("Calculating sharpness (periodic)...")
                sharpness = self.calculate_sharpness()
                print(f"Sharpness of minimum: {sharpness:.6f}")
                if self.logger.use_tensorboard:
                    self.logger.writer.add_scalar('Generalization/Sharpness', sharpness, epoch)
        
        # Generate comprehensive generalization report
        gen_report = self.gen_analyzer.generate_report(
            save_path=os.path.join(self.logger.log_dir, 'generalization_report.json')
        )
        
        # Plot generalization curves
        print("Plotting generalization curves to logger...")
        plot_dir = os.path.join(self.logger.log_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        self.gen_analyzer.plot_loss_curves(os.path.join(plot_dir, 'loss_curves.png'))
        self.gen_analyzer.plot_accuracy_curves(os.path.join(plot_dir, 'accuracy_curves.png'))
        self.gen_analyzer.plot_calibration_curve(os.path.join(plot_dir, 'calibration_curve.png'))
        
        # Add sharpness to the report
        final_sharpness = self.calculate_sharpness()
        gen_report['final_sharpness'] = final_sharpness
        with open(os.path.join(self.logger.log_dir, 'generalization_report.json'), 'w') as f:
            json.dump(gen_report, f, indent=4)
        
        print(f'Final sharpness: {final_sharpness:.6f}')
        
        # Close logger
        self.logger.close()
        
        print(f'Training completed. Best accuracy: {best_acc:.2f}%')