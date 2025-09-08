import torch, json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class Accuracy:
    """
    Computes and stores the average and current accuracy
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
        self.avg = 0
    
    def update(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()
        self.avg = 100.0 * self.correct / self.total
        
        return self.avg

class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GeneralizationAnalyzer:
    """
    Comprehensive analyzer for generalization error and related metrics
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.all_predictions = []
        self.all_targets = []
        self.all_confidences = []
        self.all_probabilities = []  # Add this line
        self.epoch_data = {}
    
    def update_epoch(self, epoch, train_loss, train_acc, test_loss, test_acc):
        """Update metrics for a complete epoch"""
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        
        self.epoch_data[epoch] = {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'generalization_gap': train_acc - test_acc,
            'overfitting_ratio': (train_acc - test_acc) / train_acc if train_acc > 0 else 0
        }
    
    def update_batch(self, outputs, targets):
        """Update batch-level predictions and confidences"""
        # Detach the outputs from the computation graph before converting to numpy
        with torch.no_grad():
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, 1)

            # Use detach() before converting to numpy
            self.all_predictions.extend(predictions.cpu().numpy())
            self.all_targets.extend(targets.cpu().numpy())
            self.all_confidences.extend(confidences.cpu().numpy())
            self.all_probabilities.extend(probs.cpu().numpy())  # Store all probabilities
    
    def calculate_generalization_gap(self):
        """Calculate the generalization gap across all epochs"""
        if not self.train_accuracies or not self.test_accuracies:
            return 0
        
        avg_train_acc = np.mean(self.train_accuracies)
        avg_test_acc = np.mean(self.test_accuracies)
        return avg_train_acc - avg_test_acc
    
    def calculate_overfitting_metric(self):
        """Calculate a metric that quantifies overfitting"""
        if not self.train_losses or not self.test_losses:
            return 0
        
        # Ratio of final test loss to minimum test loss (higher indicates more overfitting)
        min_test_loss = min(self.test_losses)
        final_test_loss = self.test_losses[-1]
        return final_test_loss / min_test_loss if min_test_loss > 0 else 0
    
    def calculate_expected_calibration_error(self):
        """Calculate Expected Calibration Error (ECE)"""
        if not self.all_confidences or not self.all_predictions or not self.all_targets:
            return 0
        
        # Convert to numpy arrays
        predictions = np.array(self.all_predictions)
        targets = np.array(self.all_targets)
        confidences = np.array(self.all_confidences)
        
        # Bin the confidences
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if the confidence falls in the bin
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                # Calculate the accuracy in this bin
                accuracy_in_bin = np.mean(predictions[in_bin] == targets[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                
                # Add to ECE
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    def calculate_auc_roc(self):
        """Calculate AUC-ROC for multi-class classification"""
        if not self.all_probabilities or not self.all_targets:
            return 0
        
        # Convert to numpy arrays and ensure they have the same length
        min_length = min(len(self.all_targets), len(self.all_probabilities))
        if min_length == 0:
            return 0
        
        targets_np = np.array(self.all_targets[:min_length])
        probabilities_np = np.array(self.all_probabilities[:min_length])
        
        # One-hot encode targets
        targets_one_hot = np.eye(self.num_classes)[targets_np]
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_one_hot[:, i], probabilities_np[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(targets_one_hot.ravel(), probabilities_np.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        return roc_auc
    
    def calculate_confidence_metrics(self):
        """Calculate various confidence-related metrics"""
        if not self.all_confidences:
            return {}
        
        confidences = np.array(self.all_confidences)
        correct = np.array(self.all_predictions) == np.array(self.all_targets)
        
        avg_confidence = np.mean(confidences)
        avg_confidence_correct = np.mean(confidences[correct]) if np.any(correct) else 0
        avg_confidence_incorrect = np.mean(confidences[~correct]) if np.any(~correct) else 0
        
        return {
            'avg_confidence': avg_confidence,
            'avg_confidence_correct': avg_confidence_correct,
            'avg_confidence_incorrect': avg_confidence_incorrect,
            'confidence_gap': avg_confidence_correct - avg_confidence_incorrect
        }
    
    def plot_calibration_curve(self, save_path=None):
        """Plot calibration curve for model predictions"""
        if not self.all_confidences or not self.all_targets:
            return
        
        # For multi-class problems, we cannot just use sklearn calibration_curve bc it expects binary
        # One common approach is to use a reliability diagram for the true class probabilities
        try:
            # Create a reliability diagram
            bin_boundaries = np.linspace(0, 1, 11)  # 10 bins
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_centers = 0.5 * (bin_lowers + bin_uppers)
            empirical_prob = np.zeros(len(bin_centers))
            predicted_prob = np.zeros(len(bin_centers))
            
            # Calculate accuracy and confidence in each bin
            for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
                in_bin = np.logical_and(self.all_confidences >= bin_lower, 
                                    self.all_confidences < bin_upper)
                if np.any(in_bin):
                    # Empirical probability (accuracy) in this bin
                    empirical_prob[i] = np.mean(np.array(self.all_predictions)[in_bin] == 
                                            np.array(self.all_targets)[in_bin])
                    # Predicted probability in this bin
                    predicted_prob[i] = np.mean(np.array(self.all_confidences)[in_bin])
            
            # Plot reliability diagram
            plt.figure(figsize=(10, 8))
            plt.plot(bin_centers, empirical_prob, marker='o', label='Model')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
            plt.xlabel('Predicted probability')
            plt.ylabel('Empirical probability')
            plt.title('Reliability diagram')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
                
        except Exception as e: # Fallback: don't plot rather than crash
            print(f"Error plotting calibration curve: {e}")
    
    def plot_loss_curves(self, save_path=None):
        """Plot training and test loss curves"""
        if not self.train_losses or not self.test_losses:
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.test_losses, label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and test loss curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_accuracy_curves(self, save_path=None):
        """Plot training and test accuracy curves"""
        if not self.train_accuracies or not self.test_accuracies:
            return
        
        plt.figure(figsize=(10, 8))
        plt.plot(self.train_accuracies, label='Training accuracy')
        plt.plot(self.test_accuracies, label='Test accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and test accuracy curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
    def generate_report(self, save_path=None):
        """Generate a comprehensive report of generalization metrics"""
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types because it wasn't working automatically"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj

        report = {
            'generalization_gap': self.calculate_generalization_gap(),
            'overfitting_metric': self.calculate_overfitting_metric(),
            'expected_calibration_error': self.calculate_expected_calibration_error(),
            'auc_roc': self.calculate_auc_roc(),
            'confidence_metrics': self.calculate_confidence_metrics(),
            'epoch_data': self.epoch_data,
            'final_sharpness': self.calculate_sharpness() if hasattr(self, 'calculate_sharpness') else None
        }
        
        # Convert all numpy types to serializable types
        report = convert_to_serializable(report)
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

class SharpnessCalculator:
    """
    Calculate sharpness of minima as a measure of generalization
    Based on: Keskar et al. "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"
    """
    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon
        self.original_params = {}
        
    def calculate_sharpness(self, loss_fn, data_loader, device):
        """Calculate sharpness of the current minimum"""
        # Save original parameters
        self._save_parameters()
        
        # Calculate original loss
        original_loss = self._calculate_loss(loss_fn, data_loader, device)
        
        # Perturb parameters and calculate maximum loss increase
        max_loss_increase = 0
        
        for param in self.model.parameters():
            if param.requires_grad:
                # Save original parameter values
                original_values = param.data.clone()
                
                # Perturb in positive direction
                param.data.add_(self.epsilon)
                loss_plus = self._calculate_loss(loss_fn, data_loader, device)
                increase_plus = max(0, loss_plus - original_loss) / (1 + original_loss)
                
                # Perturb in negative direction
                param.data.copy_(original_values)
                param.data.add_(-self.epsilon)
                loss_minus = self._calculate_loss(loss_fn, data_loader, device)
                increase_minus = max(0, loss_minus - original_loss) / (1 + original_loss)
                
                # Restore original values
                param.data.copy_(original_values)
                
                # Update maximum loss increase
                max_loss_increase = max(max_loss_increase, increase_plus, increase_minus)
        
        # Restore original parameters
        self._restore_parameters()
        
        return max_loss_increase
    
    def _save_parameters(self):
        """Save current model parameters"""
        self.original_params = {
            name: param.data.clone() for name, param in self.model.named_parameters()
        }
    
    def _restore_parameters(self):
        """Restore saved model parameters"""
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                param.data.copy_(self.original_params[name])
    
    def _calculate_loss(self, loss_fn, data_loader, device):
        """Calculate loss on the given data loader"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        
        return total_loss / total_samples if total_samples > 0 else 0