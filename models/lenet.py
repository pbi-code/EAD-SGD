import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    LeNet-5 architecture for MNIST dataset
    Reference: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # Register hooks for K-FAC
        self.activations = {}
        self.gradients = {}
        
    def forward(self, x):
        # Layer 1: Convolution -> ReLU -> AvgPool
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # Layer 2: Convolution -> ReLU -> AvgPool
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                # For register_full_backward_hook, grad_output is a tuple
                self.gradients[name] = grad_output[0].detach() if grad_output[0] is not None else None
            return hook
        
        # Register hooks for each layer
        self.conv1.register_forward_hook(get_activation('conv1'))
        self.conv2.register_forward_hook(get_activation('conv2'))
        self.fc1.register_forward_hook(get_activation('fc1'))
        self.fc2.register_forward_hook(get_activation('fc2'))
        
        # Use register_full_backward_hook instead of register_backward_hook
        self.conv1.register_full_backward_hook(get_gradient('conv1'))
        self.conv2.register_full_backward_hook(get_gradient('conv2'))
        self.fc1.register_full_backward_hook(get_gradient('fc1'))
        self.fc2.register_full_backward_hook(get_gradient('fc2'))