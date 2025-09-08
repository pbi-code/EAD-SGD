import os
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from .metrics import GeneralizationAnalyzer

class Logger:
    """
    Enhanced logger for tracking training progress and generalization metrics
    """
    def __init__(self, log_dir, use_tensorboard=True, num_classes=10, optimizer=None):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.num_classes = num_classes
        self.optimizer = optimizer
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer if requested
        if use_tensorboard:
            tb_dir = os.path.join(log_dir, 'tensorboard')
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(tb_dir)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.log_data = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_times': [],
            'best_acc': 0.0,
            'best_epoch': 0,
            'generalization_metrics': {}
        }
        
        # Initialize generalization analyzer
        self.analyzer = GeneralizationAnalyzer(num_classes)

        # Initialize k_sparsity list if using EASGD
        if optimizer is not None and hasattr(optimizer, 'get_current_k_sparsity'):
            self.log_data['k_sparsity'] = []
        else:
            self.log_data['k_sparsity'] = None
    
    def log_metrics(self, epoch, train_loss, train_acc, test_loss, test_acc, epoch_time):
        """Log metrics for a single epoch and update generalization analyzer"""
        # Update internal log
        self.log_data['train_loss'].append(train_loss)
        self.log_data['train_acc'].append(train_acc)
        self.log_data['test_loss'].append(test_loss)
        self.log_data['test_acc'].append(test_acc)
        self.log_data['epoch_times'].append(epoch_time)
        
        # Update generalization analyzer
        self.analyzer.update_epoch(epoch, train_loss, train_acc, test_loss, test_acc)
        
        # Update best accuracy
        if test_acc > self.log_data['best_acc']:
            self.log_data['best_acc'] = test_acc
            self.log_data['best_epoch'] = epoch
        
        # Log k-sparsity only if optimizer is available and has the method (EASGD with dynamic scheduling)
        if self.optimizer is not None and hasattr(self.optimizer, 'get_current_k_sparsity'):
            k_sparsity = self.optimizer.get_current_k_sparsity()
            if self.log_data['k_sparsity'] is not None:
                self.log_data['k_sparsity'].append(k_sparsity)
        
        # Write to TensorBoard
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Loss/Test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/Test', test_acc, epoch)
            self.writer.add_scalar('Time/Epoch', epoch_time, epoch)
            
            # Log generalization gap
            gen_gap = train_acc - test_acc
            self.writer.add_scalar('Generalization/Gap', gen_gap, epoch)
            
            # Log overfitting ratio
            overfitting_ratio = (train_acc - test_acc) / train_acc if train_acc > 0 else 0
            self.writer.add_scalar('Generalization/OverfittingRatio', overfitting_ratio, epoch)

            # Log k-sparsity if using EASGD with dynamic scheduling
            self.writer.add_scalar('K-Sparsity/Current', k_sparsity, epoch)
        
        # Save to JSON file
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
    
    def log_batch_predictions(self, outputs, targets):
        """Log batch predictions for generalization analysis"""
        self.analyzer.update_batch(outputs, targets)
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to TensorBoard"""
        if self.use_tensorboard:
            # Convert config to a format TensorBoard can handle
            from torch.utils.tensorboard.summary import hparams
            hparam_dict = {k: v for k, v in config.items() if not isinstance(v, dict)}
            metric_dict = {'hp/best_accuracy': 0}  # Will be updated later
            
            # Add nested config values
            for section, section_config in config.items():
                if isinstance(section_config, dict):
                    for k, v in section_config.items():
                        hparam_dict[f"{section}/{k}"] = v
            
            self.writer.add_hparams(hparam_dict, metric_dict)
    
    def generate_generalization_report(self, save_path=None):
        """Generate a comprehensive generalization report"""
        report = self.analyzer.generate_report(save_path)
        self.log_data['generalization_metrics'] = report
        
        # Save updated log data
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)
        
        return report
    
    def plot_generalization_curves(self, save_dir=None):
        """Plot generalization-related curves"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Plot loss curves
            loss_path = os.path.join(save_dir, 'loss_curves.png')
            self.analyzer.plot_loss_curves(loss_path)
            
            # Plot accuracy curves
            acc_path = os.path.join(save_dir, 'accuracy_curves.png')
            self.analyzer.plot_accuracy_curves(acc_path)
            
            # Plot calibration curve
            cal_path = os.path.join(save_dir, 'calibration_curve.png')
            self.analyzer.plot_calibration_curve(cal_path)
        else:
            self.analyzer.plot_loss_curves()
            self.analyzer.plot_accuracy_curves()
            self.analyzer.plot_calibration_curve()
    
    def close(self):
        """Close the logger and all writers"""
        if self.use_tensorboard:
            self.writer.close()