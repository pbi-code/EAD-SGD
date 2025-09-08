import math
import torch

class KSparsityScheduler:
    """
    Dynamic k-sparsity scheduler for EASGD optimizer
    Adjusts the fraction of components to keep in WHT domain during training
    """
    
    def __init__(self, initial_k=0.5, final_k=0.1, total_steps=1000, 
                 strategy='linear', warmup_steps=100, cycle_length=500):
        """
        Initialize the k-sparsity scheduler
        
        Args:
            initial_k: Initial fraction of components to keep (0.0 to 1.0)
            final_k: Final fraction of components to keep (0.0 to 1.0)
            total_steps: Total training steps for the schedule
            strategy: Scheduling strategy ('linear', 'exponential', 'cosine', 'step', 'cyclic')
            warmup_steps: Number of warmup steps before starting decay
            cycle_length: Length of cycle for cyclic scheduling
        """
        self.initial_k = initial_k
        self.final_k = final_k
        self.total_steps = total_steps
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.current_step = 0
        
        # Validate inputs
        assert 0.0 <= initial_k <= 1.0, "initial_k must be between 0.0 and 1.0"
        assert 0.0 <= final_k <= 1.0, "final_k must be between 0.0 and 1.0"
        assert initial_k >= final_k, "initial_k should be >= final_k for effective pruning"
        
    def step(self):
        """Update the current step and return the current k-sparsity"""
        self.current_step += 1
        return self.get_k_sparsity()
    
    def get_k_sparsity(self):
        """Get the current k-sparsity value based on the scheduling strategy"""
        if self.current_step < self.warmup_steps:
            # Warmup phase: use initial_k
            return self.initial_k
        
        effective_step = self.current_step - self.warmup_steps
        progress = min(effective_step / (self.total_steps - self.warmup_steps), 1.0)
        
        if self.strategy == 'linear':
            k = self.initial_k - (self.initial_k - self.final_k) * progress
            
        elif self.strategy == 'exponential':
            # Exponential decay: k = initial_k * (final_k/initial_k)^progress
            if self.initial_k == 0:
                k = 0
            else:
                k = self.initial_k * (self.final_k / self.initial_k) ** progress
                
        elif self.strategy == 'cosine':
            # Cosine annealing
            k = self.final_k + 0.5 * (self.initial_k - self.final_k) * (
                1 + math.cos(math.pi * progress))
                
        elif self.strategy == 'step':
            # Step decay: reduce by half every 25% of total steps
            num_steps = 4  # Reduce every 25% of steps
            step_size = (self.total_steps - self.warmup_steps) / num_steps
            step = int(effective_step / step_size)
            k = self.initial_k * (0.5 ** step)
            k = max(k, self.final_k)
            
        elif self.strategy == 'cyclic':
            # Cyclic scheduling: oscillate between initial_k and final_k
            cycle_progress = (effective_step % self.cycle_length) / self.cycle_length
            k = self.final_k + 0.5 * (self.initial_k - self.final_k) * (
                1 + math.cos(2 * math.pi * cycle_progress))
                
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")
        
        return max(k, self.final_k)  # Ensure k doesn't go below final_k
    
    def state_dict(self):
        """Return the state of the scheduler"""
        return {
            'current_step': self.current_step,
            'initial_k': self.initial_k,
            'final_k': self.final_k,
            'total_steps': self.total_steps,
            'strategy': self.strategy,
            'warmup_steps': self.warmup_steps,
            'cycle_length': self.cycle_length
        }
    
    def load_state_dict(self, state_dict):
        """Load the state of the scheduler"""
        self.current_step = state_dict['current_step']
        self.initial_k = state_dict['initial_k']
        self.final_k = state_dict['final_k']
        self.total_steps = state_dict['total_steps']
        self.strategy = state_dict['strategy']
        self.warmup_steps = state_dict['warmup_steps']
        self.cycle_length = state_dict['cycle_length']