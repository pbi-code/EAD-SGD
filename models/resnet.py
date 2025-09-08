import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        # Register hooks for K-FAC
        self.activations = {}
        self.gradients = {}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
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
        self.layer1.register_forward_hook(get_activation('layer1'))
        self.layer2.register_forward_hook(get_activation('layer2'))
        self.layer3.register_forward_hook(get_activation('layer3'))
        self.layer4.register_forward_hook(get_activation('layer4'))
        
        # Use register_full_backward_hook instead of register_backward_hook
        self.conv1.register_full_backward_hook(get_gradient('conv1'))
        self.layer1.register_full_backward_hook(get_gradient('layer1'))
        self.layer2.register_full_backward_hook(get_gradient('layer2'))
        self.layer3.register_full_backward_hook(get_gradient('layer3'))
        self.layer4.register_full_backward_hook(get_gradient('layer4'))

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])