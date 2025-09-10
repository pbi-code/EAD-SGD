import yaml
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
import os, sys, gc
import time, random, torch
from collections import defaultdict
from tqdm import tqdm

# Configuration
database = "mnist"
# optimizers = ["sgd", "sgdm", "adam", "entropysgd", "easgd"]
# optimizers = ["sgd", "sgdm", "adam", "easgd"]
optimizers = ["sgd", "sgdm"]

# Produce logarithmically-spaced epochs array
a1 = 10.**np.arange(1, 3)
a2 = np.arange(1, 10, 2)
epochs_range = np.outer(a1, a2).astype(np.int64).flatten().tolist()
# epochs_range = list(range(10, 101, 10))  # 10 to 100 in steps of 10
num_runs = 3  # Number of runs per configuration
print(f"Your experiment will compare optimizers {optimizers} during task {database}")
print(f"Your experiment will run each optimizer for {epochs_range} epochs with {num_runs} runs each")
input("Are you ok with these settings? Press Enter to continue...")

# Results storage
results = defaultdict(lambda: defaultdict(list))

def run_experiment(optimizer_name, epochs, run_id, pbar=None):
    """Run a single experiment with modified epochs"""
    # Determine config path
    config_path = f"configs/{database}_{optimizer_name}.yaml"
    
    # Load the base config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds for reproducibility
    seed = run_id  # or use a fixed seed per run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Update progress bar description
    if pbar:
        pbar.set_description(f"Running {optimizer_name}-{epochs}epochs-run{run_id}")
    
    # Create a unique experiment name
    # optimizer_name = os.path.basename(config_path).replace(f"{database}_", '').replace('.yaml', '')
    experiment_name = f"sharpness_{database}_{optimizer_name}_{epochs}epochs_run{run_id}"
    
    # Update the config
    config['training']['epochs'] = epochs
    config['logging']['log_dir'] = f"./logs/{experiment_name}"
    
    # Save the modified config
    temp_config_path = f"configs/temp_{experiment_name}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run the experiment
    print(f"Running {experiment_name}...")
    try:
        print("trying subprocess...")
        result = subprocess.run([
            'python', 'main.py', '--silent', '--config', temp_config_path
        ], capture_output=True, text=True, timeout=1800)  # 30-minute timeout

        print(f"Subprocess completed with return code: {result.returncode}")

        if result.returncode != 0:
            print(f"Error running {experiment_name}: {result.stderr}")
            # Also print stdout to see what happened
            print(f"STDOUT: {result.stdout}")
            return None, None
        
        # Extract results
        report_path = os.path.join(config['logging']['log_dir'], 'generalization_report.json')
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = json.load(f)
            return report.get('final_sharpness'), report.get('test_accuracy')
        else:
            print(f"Report not found for {experiment_name}")
            # Check if the log directory exists and what's in it
            log_dir = config['logging']['log_dir']
            if os.path.exists(log_dir):
                print(f"Contents of {log_dir}: {os.listdir(log_dir)}")
            return None, None
            
    except subprocess.TimeoutExpired:
        print(f"Experiment {experiment_name} timed out after 30 minutes")
        return None, None
    except Exception as e:
        print(f"Error running {experiment_name}: {e}")
        # Log the error to a file for later analysis
        with open("failed_experiments.log", "a") as log_file:
            log_file.write(f"{experiment_name}: {str(e)}\n")
        return None, None
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        # Update progress bar
        if pbar:
            pbar.update(1)

def main():
    # Calculate total number of experiments
    total_experiments = len(optimizers) * len(epochs_range) * num_runs
    
    # Create progress bar
    with tqdm(total=total_experiments, desc="Overall Progress", unit="exp") as pbar:
        # Run all experiments
        for optimizer in optimizers:
            if not os.path.exists(f"configs/{database}_{optimizer}.yaml"):
                print(f"Config file not found: configs/{database}_{optimizer}.yaml")
                # Update progress bar for skipped experiments
                pbar.update(len(epochs_range) * num_runs)
                continue
                
            for epochs in epochs_range:
                for run in range(1, num_runs + 1):
                    sharpness, accuracy = run_experiment(optimizer, epochs, run, pbar)
                    
                    if sharpness is not None and accuracy is not None:
                        results[optimizer][epochs].append((sharpness, accuracy))
                    
                    # Add a small delay between runs
                    time.sleep(2)
    
    # Save results
    with open('sharpness_results.json', 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for optimizer, epoch_data in results.items():
            serializable_results[optimizer] = {}
            for epoch, values in epoch_data.items():
                serializable_results[optimizer][epoch] = [
                    (float(sharpness), float(accuracy)) for sharpness, accuracy in values
                ]
        json.dump(serializable_results, f, indent=4)
    
    # Generate plots
    generate_plots(results)

def generate_plots(results):
    """Generate plots from the results"""
    # Plot 1: Sharpness vs Epochs for each optimizer
    plt.figure(figsize=(12, 5))
    
    # Sharpness plot
    plt.subplot(1, 2, 1)
    for optimizer, epoch_data in results.items():
        epochs = sorted(epoch_data.keys())
        sharpness_means = []
        sharpness_stds = []
        
        for epoch in epochs:
            values = epoch_data[epoch]
            sharpness_vals = [v[0] for v in values]
            sharpness_means.append(np.mean(sharpness_vals))
            sharpness_stds.append(np.std(sharpness_vals))
        
        plt.errorbar(epochs, sharpness_means, yerr=sharpness_stds, 
                    marker='o', capsize=5, label=optimizer)
    
    plt.xlabel('Epochs')
    plt.ylabel('Sharpness')
    plt.title('Sharpness vs Training Epochs')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    for optimizer, epoch_data in results.items():
        epochs = sorted(epoch_data.keys())
        accuracy_means = []
        accuracy_stds = []
        
        for epoch in epochs:
            values = epoch_data[epoch]
            accuracy_vals = [v[1] for v in values]
            accuracy_means.append(np.mean(accuracy_vals))
            accuracy_stds.append(np.std(accuracy_vals))
        
        plt.errorbar(epochs, accuracy_means, yerr=accuracy_stds, 
                    marker='o', capsize=5, label=optimizer)
    
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Training Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sharpness_accuracy_vs_epochs.png')
    plt.show()
    
    # Plot 2: Accuracy vs Sharpness (for the final epoch)
    plt.figure(figsize=(8, 6))
    final_epoch = max(epochs_range)
    
    for optimizer, epoch_data in results.items():
        if final_epoch in epoch_data:
            values = epoch_data[final_epoch]
            sharpness_vals = [v[0] for v in values]
            accuracy_vals = [v[1] for v in values]
            
            plt.scatter(sharpness_vals, accuracy_vals, label=optimizer, alpha=0.7)
    
    plt.xlabel('Sharpness')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Accuracy vs Sharpness at {final_epoch} Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_sharpness.png')
    plt.show()

if __name__ == "__main__":
    main()