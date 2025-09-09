import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Optional
from hadamard_transform import hadamard_transform
from utils.k_sparsity_scheduler import KSparsityScheduler
import math as m

class EA_SGD(torch.optim.Optimizer):
    """
    Anisotropic Entropic-SGD with Dual Information Projections
    
    This optimizer implements the EA-SGD algorithm which combines:
    1. Local entropy exploration via SGLD (M-projection)
    2. Information-rich subspace selection via SNR-based WHT compression (E-projection)
    3. K-FAC approximation for Fisher Information Matrix preconditioning
    
    Reference: Based on the information geometry principles from Nielsen's work
    and the Entropy-SGD approach from Chaudhari et al.
    """
    
    def __init__(self, params, lr=0.01, inner_lr=0.1, inner_steps=20, thermal_noise=1e-3, 
             projection_freq=5, k_sparsity=0.1, gamma=0.03, 
             momentum=0.9, weight_decay=0, kfac_update_freq=100, k_sparsity_schedule=None):
        """
        Initialize the EA-SGD optimizer
        
        Args:
            params: Model parameters to optimize
            lr: Learning rate for the outer optimization loop
            inner_lr: Learning rate for the inner SGLD sampling
            inner_steps: Number of SGLD steps for M-projection
            projection_freq: Frequency (in steps) of performing M and E projections
            k_sparsity: Fraction of components to keep in the WHT domain (0-1)
            gamma: Strength of the local entropy regularizer
            momentum: Momentum factor
            weight_decay: Weight decay (L2 penalty)
            kfac_update_freq: Frequency of updating K-FAC matrices
            k_sparsity_schedule: Dictionary with k-sparsity scheduling parameters
        """
        defaults = dict(lr=lr, inner_lr=inner_lr, inner_steps=inner_steps, thermal_noise=thermal_noise,
                   projection_freq=projection_freq, k_sparsity=k_sparsity,
                   gamma=gamma, momentum=momentum, weight_decay=weight_decay,
                   kfac_update_freq=kfac_update_freq)
        
        super(EA_SGD, self).__init__(params, defaults)
        self.is_ea_sgd = True # identifier
        
        # Initialize state variables
        self.state = defaultdict(dict)
        self.step_count = 0
        
        # Initialize K-FAC matrices for each parameter group
        self._init_kfac_matrices()
        
        # Initialize WHT mask (will be updated during E-projection)
        self.wht_mask = None
        self.projection_freq = projection_freq

        self.gamma = gamma
        self.initial_gamma = gamma
        self.gamma_growth_rate = 1.001  # Increase gamma by 0.1% each step

        # Pre-allocate noise buffers for SGLD
        self.noise_buffers = []
        for group in self.param_groups:
            for p in group['params']:
                self.noise_buffers.append(torch.empty_like(p.data))

        # Initialize k-sparsity scheduler
        if k_sparsity_schedule is None:
            # Default schedule: linear decay from 0.5 to 0.1 over 1000 steps
            k_sparsity_schedule = {
                'initial_k': 0.5,
                'final_k': 0.1,
                'total_steps': 1000,
                'strategy': 'linear',
                'warmup_steps': 100
            }
        
        self.k_scheduler = KSparsityScheduler(**k_sparsity_schedule)
        
    def _init_kfac_matrices(self):
        """Initialize K-FAC matrices for Fisher Information approximation"""
        for group in self.param_groups:
            for p in group['params']:
                if p.dim() >= 2:  # Only for weight matrices, not biases
                    state = self.state[p]
                    # A matrix (activation covariance)
                    state['A'] = torch.eye(p.size(1), device=p.device) * 1e-3
                    # G matrix (gradient covariance)
                    state['G'] = torch.eye(p.size(0), device=p.device) * 1e-3
                    # Inverse matrices (will be computed periodically)
                    state['A_inv'] = torch.eye(p.size(1), device=p.device)
                    state['G_inv'] = torch.eye(p.size(0), device=p.device)
    
    def _update_kfac_matrices(self, activations: dict, gradients: dict):
        """
        Update K-FAC matrices with current activations and gradients
        
        Args:
            activations: Dictionary of layer activations from forward pass
            gradients: Dictionary of layer gradients from backward pass
        """
        beta = 0.95  # Exponential moving average factor
        
        for group in self.param_groups:
            for p in group['params']:
                if p in activations and p in gradients:
                    state = self.state[p]
                    a = activations[p]
                    g = gradients[p]
                    
                    # Update A matrix (activation covariance)
                    if a.dim() > 2:  # For convolutional layers
                        a = a.flatten(2)  # Flatten spatial dimensions
                    
                    # Compute outer product and update with EMA
                    a_outer = torch.einsum('bi,bj->ij', a, a) / a.size(0)
                    state['A'] = beta * state['A'] + (1 - beta) * a_outer
                    
                    # Update G matrix (gradient covariance)
                    if g.dim() > 2:  # For convolutional layers
                        g = g.flatten(2)  # Flatten spatial dimensions
                    
                    g_outer = torch.einsum('bi,bj->ij', g, g) / g.size(0)
                    state['G'] = beta * state['G'] + (1 - beta) * g_outer
                    
                    # Update inverses if it's time
                    if self.step_count % group['kfac_update_freq'] == 0:
                        state['A_inv'] = torch.inverse(state['A'] + 1e-6 * torch.eye(state['A'].size(0), device=p.device))
                        state['G_inv'] = torch.inverse(state['G'] + 1e-6 * torch.eye(state['G'].size(0), device=p.device))
    
    # def _precondition_gradient(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    #     """
    #     Precondition gradient using K-FAC approximation
        
    #     Args:
    #         p: Parameter tensor
    #         g: Gradient tensor
            
    #     Returns:
    #         Preconditioned gradient
    #     """
    #     state = self.state[p]
        
    #     if p.dim() == 1:  # Bias term
    #         return g
        
    #     # For 2D weights: G_inv * g * A_inv
    #     if p.dim() == 2:
    #         return state['G_inv'] @ g @ state['A_inv']
        
    #     # For 4D conv weights: reshape and apply K-FAC
    #     elif p.dim() == 4:
    #         # Reshape to 2D for K-FAC
    #         g_2d = g.view(g.size(0), -1)
    #         pre_g_2d = state['G_inv'] @ g_2d @ state['A_inv']
    #         return pre_g_2d.view_as(g)
        
    #     return g  # Fallback for other cases
    
    def _precondition_gradient(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Precondition gradient using K-FAC approximation
        """
        # Skip preconditioning if KFAC is not being updated
        if self.param_groups[0]['kfac_update_freq'] > 10000:
            return g
        
        state = self.state[p]
        
        # Only precondition linear layers (2D weights)
        if p.dim() == 2 and 'G_inv' in state and 'A_inv' in state:
            return state['G_inv'] @ g @ state['A_inv']
        
        # For all other layers (convolutional, biases), return the original gradient
        return g
    
    def _m_projection(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                     group: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform M-projection via SGLD sampling of the local entropy landscape
        
        Args:
            model: The neural network model
            data_loader: Data loader for sampling batches
            group: Parameter group dictionary
            
        Returns:
            mean_grad: Mean of the sampled gradients
            cov_grad: Covariance of the sampled gradients (diagonal approximation)
        """
        # Store original parameters
        original_params = [p.detach().clone() for p in model.parameters()]

        # Create a persistent iterator
        data_iter = iter(data_loader)
        
        # Initialize lists to store gradients
        all_gradients = []
        
        # Perform SGLD sampling
        for step in range(group['inner_steps']):
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                data, target = next(data_iter)
            
            data, target = data.to(self.device), target.to(self.device)
        
            # Forward pass
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            
            # Add local entropy regularizer
            for i, p in enumerate(model.parameters()):
                if p.grad is not None:
                    p.grad.add_(group['gamma'] * (p.data - original_params[i].data))
            
            # Compute gradient
            loss.backward()
            
            # Store gradient
            gradients = []
            for p in model.parameters():
                if p.grad is not None:
                    gradients.append(p.grad.detach().clone().flatten())
            
            all_gradients.append(torch.cat(gradients))
            
            # SGLD update: gradient step + noise
            with torch.no_grad():
                for i, p in enumerate(model.parameters()):
                    if p.grad is not None:
                        # Use pre-allocated buffer
                        self.noise_buffers[i].normal_()
                        noise = self.noise_buffers[i] * np.sqrt(2 * group['inner_lr']) * group['thermal_noise']
                        p.data -= group['inner_lr'] * p.grad + noise
            
            # Zero gradients for next step
            model.zero_grad()
        
        # Restore original parameters
        with torch.no_grad():
            for p, orig in zip(model.parameters(), original_params):
                p.data.copy_(orig)
        
        # Compute statistics of sampled gradients
        all_gradients = torch.stack(all_gradients)
        mean_grad = all_gradients.mean(dim=0)
        cov_grad = all_gradients.var(dim=0)  # Diagonal approximation
        
        return mean_grad, cov_grad

    def _next_power_of_2(self, n):
        """Find the next power of 2 greater than or equal to n"""
        n -= 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n += 1
        return n

    def _pad_to_power_of_2(self, x):
        """Pad a vector to the next power of 2 length"""
        n = x.shape[0]
        next_power = self._next_power_of_2(n)
        if n == next_power:
            return x, n
        else:
            padded = torch.zeros(next_power, dtype=x.dtype, device=x.device)
            padded[:n] = x
            return padded, n
    
    # def _e_projection(self, mean_grad: torch.Tensor, cov_grad: torch.Tensor, 
    #              group: dict) -> torch.Tensor:
    #     """
    #     Perform E-projection via SNR-based component selection in WHT domain
        
    #     Args:
    #         mean_grad: Mean of gradients from M-projection
    #         cov_grad: Variance of gradients from M-projection
    #         group: Parameter group dictionary
            
    #     Returns:
    #         mask: Binary mask for WHT components
    #     """
    #     # Pad to power of 2
    #     mean_grad_padded, orig_len = self._pad_to_power_of_2(mean_grad)
    #     cov_grad_padded, _ = self._pad_to_power_of_2(cov_grad)
        
    #     # Apply FWHT
    #     wht_mean = hadamard_transform(mean_grad_padded)
    #     wht_cov = hadamard_transform(cov_grad_padded)
        
    #     # Compute SNR for each component - FWHT outputs real numbers
    #     snr = torch.abs(wht_mean) / (torch.sqrt(torch.abs(wht_cov)) + 1e-8)
        
    #     # Select top-k components based on SNR
    #     k = int(group['k_sparsity'] * len(snr))
    #     _, topk_indices = torch.topk(snr, k)
        
    #     # Create mask of padded length
    #     mask = torch.zeros_like(mean_grad_padded, dtype=torch.bool)
    #     mask[topk_indices] = True
        
    #     return mask
        
    def _e_projection(self, mean_grad: torch.Tensor, cov_grad: torch.Tensor, group: dict) -> List[torch.Tensor]:
        """
        Perform E-projection via SNR-based component selection in WHT domain.
        Returns a list of masks for each parameter.
        """
        # Split the mean_grad and cov_grad into per-parameter segments
        param_shapes = [p.numel() for p in self._model.parameters()]
        mean_grads = torch.split(mean_grad, param_shapes)
        cov_grads = torch.split(cov_grad, param_shapes)
        
        masks = []
        for mg, cg in zip(mean_grads, cov_grads):
            # Pad to power of 2
            mg_padded, orig_len = self._pad_to_power_of_2(mg)
            cg_padded, _ = self._pad_to_power_of_2(cg)
            
            # Apply FWHT
            wht_mean = hadamard_transform(mg_padded)
            wht_cov = hadamard_transform(cg_padded)
            
            # Compute SNR for each component
            snr = torch.abs(wht_mean) / (torch.sqrt(torch.abs(wht_cov)) + 1e-8)
            
            # Select top-k components based on SNR
            k = int(group['k_sparsity'] * len(snr))
            _, topk_indices = torch.topk(snr, k)
            
            # Create mask
            mask = torch.zeros_like(mg_padded, dtype=torch.bool)
            mask[topk_indices] = True
            masks.append(mask)
        
        return masks

    def _apply_wht_mask(self, gradient: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply WHT mask to gradient with dynamic size handling
        """
        if mask is None:
            return gradient
        
        # Ensure mask is a tensor with the same device as gradient
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=gradient.device, dtype=torch.bool)

        # Flatten the gradient
        flat_grad = gradient.flatten()
        grad_len = flat_grad.shape[0]
        
        # Pad to power of 2 if needed
        padded_grad, orig_len = self._pad_to_power_of_2(flat_grad)
        
        # Apply WHT
        wht_grad = hadamard_transform(padded_grad)
        
        # Ensure mask matches the padded size
        if mask.numel() != padded_grad.numel():
        # if mask.shape[0] != padded_grad.shape[0]:
            # replace with identity if we have a size match issue
            mask = torch.eye(padded_grad.shape[0], dtype=torch.int)
            print(f"Warning: Mask size {mask.numel()} does not match padded gradient size {padded_grad.numel()}. Using identity mask.")
            # # Resize mask to match the current gradient size
            # resized_mask = torch.zeros_like(padded_grad, dtype=torch.bool)
            # min_size = min(mask.shape[0], padded_grad.shape[0])
            # resized_mask[:min_size] = mask[:min_size]
        else:
            resized_mask = mask
        
        # Apply mask
        filtered_wht = wht_grad * resized_mask
        
        # Inverse WHT
        filtered_grad_padded = hadamard_transform(filtered_wht)
        
        # Unpad and reshape
        filtered_grad = filtered_grad_padded[:grad_len]
        return filtered_grad.view_as(gradient)

    
    def step(self, closure: Optional[callable] = None, 
         data_loader: Optional[torch.utils.data.DataLoader] = None,
         activations: Optional[dict] = None,
         gradients: Optional[dict] = None) -> Optional[float]:
        """
        Perform a single optimization step
        """
        if closure is not None:
            loss = closure()
        else:
            loss = None

        # Update k-sparsity and gamma for this step
        current_k_sparsity = self.k_scheduler.step()
        self.gamma *= self.gamma_growth_rate
        for group in self.param_groups:
            group['k_sparsity'] = current_k_sparsity
            group['gamma'] = self.gamma

        # Update K-FAC matrices if activations and gradients are provided
        if activations is not None and gradients is not None:
            self._update_kfac_matrices(activations, gradients)     

        # Perform M and E projections at the specified frequency
        projection_freq = self.projection_freq  # Accessed from the instance attribute
        if self.step_count % projection_freq == 0 and data_loader is not None:
            mean_grad, cov_grad = self._m_projection(self._model, data_loader, self.param_groups[0])
            self.wht_masks = self._e_projection(mean_grad, cov_grad, self.param_groups[0])
        else:
            # If not projection step, use existing masks or set to None
            if not hasattr(self, 'wht_masks'):
                self.wht_masks = None

        # Iterate over all parameter groups and parameters
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue

                # Precondition gradient with K-FAC
                g = self._precondition_gradient(p, p.grad.data)

                # Apply WHT mask if available
                if self.wht_masks is not None and i < len(self.wht_masks):
                    mask = self.wht_masks[i]
                    if mask is not None:
                        g = self._apply_wht_mask(g, mask)

                # Apply weight decay
                if group['weight_decay'] != 0:
                    g.add_(p.data, alpha=group['weight_decay'])

                # Apply momentum
                if 'momentum_buffer' not in self.state[p]:
                    self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)
                buf = self.state[p]['momentum_buffer']
                buf.mul_(group['momentum']).add_(g)
                g = buf  # Use the momentum-adjusted gradient

                # Update parameters
                p.data.add_(g, alpha=-group['lr'])

        self.step_count += 1
        return loss
    
    def get_current_k_sparsity(self):
        """Get the current k-sparsity value"""
        return self.k_scheduler.get_k_sparsity()
    
    def state_dict(self):
        """Return the state of the optimizer including the k-scheduler"""
        state = super(EA_SGD, self).state_dict()
        state['k_scheduler'] = self.k_scheduler.state_dict()
        state['step_count'] = self.step_count
        return state
    
    def load_state_dict(self, state_dict):
        """Load the state of the optimizer including the k-scheduler"""
        super(EA_SGD, self).load_state_dict(state_dict)
        if 'k_scheduler' in state_dict:
            self.k_scheduler.load_state_dict(state_dict['k_scheduler'])
        if 'step_count' in state_dict:
            self.step_count = state_dict['step_count']

    def set_model(self, model: nn.Module):
        """Set the model reference for M-projection"""
        self._model = model
        self.device = next(model.parameters()).device