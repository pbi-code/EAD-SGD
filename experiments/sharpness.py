# sharpness.py
import yaml, json, os, time, random, subprocess, pathlib, numpy as np, matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from utils.shared_resources import init_shared_resources   # <-- NEW
from main import run_experiment                               # <-- our refactored entry point

# ----------------------------------------------------------------------
# Static configuration (you can move this to a separate yaml if you wish)
# ----------------------------------------------------------------------
DATABASE = "mnist"
OPTIMIZERS = ["sgd", "sgdm"]                     # keep your list here
# epochs_range = np.logspace(1, 2, num=5, base=10).astype(int)  # example
a1 = 10.**np.arange(1, 3)                        # 10, 100
a2 = np.arange(1, 10, 2)                        # 1,3,5,7,9
EPOCHS_RANGE = np.outer(a1, a2).astype(np.int64).flatten().tolist()
NUM_RUNS = 3

# ----------------------------------------------------------------------
# Helper that builds the *temporary* config for a given optimizer/epoch count.
# ----------------------------------------------------------------------
def build_temp_config(base_cfg_path: str, optimizer_name: str, epochs: int) -> str:
    """Writes a tiny temporary yaml that overrides epochs & log_dir."""
    with open(base_cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["epochs"] = epochs
    exp_name = f"sharpness_{DATABASE}_{optimizer_name}_{epochs}epochs_run{int(time.time()*1000)%1_000_000}"
    cfg["logging"]["log_dir"] = f"./logs/{exp_name}"
    tmp_path = f"configs/temp_{exp_name}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return tmp_path


# ----------------------------------------------------------------------
# Main driver – creates shared resources, launches the pool, aggregates.
# ----------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # 1️⃣  Initialise heavy objects once (device + data loaders)
    # --------------------------------------------------------------
    # We pick *any* config file that matches the database – here we use the first optimizer's yaml.
    base_cfg_path = f"configs/{DATABASE}_sgd.yaml"   # assumes a generic sgd yaml exists
    init_shared_resources(base_cfg_path)             # fills utils.shared_resources._shared

    # --------------------------------------------------------------
    # 2️⃣  Prepare the list of jobs for the pool
    # --------------------------------------------------------------
    jobs = []   # each element is (config_path, repeat=True, silent=True)
    for opt in OPTIMIZERS:
        cfg_path = f"configs/{DATABASE}_{opt}.yaml"
        if not os.path.exists(cfg_path):
            print(f"[WARN] Config missing: {cfg_path} – skipping.")
            continue
        for epochs in EPOCHS_RANGE:
            for run_id in range(1, NUM_RUNS + 1):
                tmp_cfg = build_temp_config(cfg_path, opt, epochs)
                jobs.append((tmp_cfg, True, True))   # (config_path, repeat, silent)

    # --------------------------------------------------------------
    # 3️⃣  Multiprocessing pool – each worker re‑uses the shared objects.
    # --------------------------------------------------------------
    # Number of processes = number of physical cores (or you can set env var OMP_NUM_THREADS)
    num_procs = min(mp.cpu_count(), 8)   # cap at 8 to avoid oversubscription on modest machines
    with mp.Pool(processes=num_procs, initializer=lambda: None) as pool:
        # tqdm wrapper around imap_unordered gives us live progress
        results = {}
        with tqdm(total=len(jobs), desc="Overall progress", unit="exp") as pbar:
            for (cfg_path, repeat, silent), res in zip(jobs, pool.imap_unordered(
                lambda args: run_experiment(*args), [(c, repeat, silent) for c, repeat, silent in jobs]
            )):
                # ``res`` is the dict returned by run_experiment
                results[cfg_path] = res
                pbar.update(1)

    # --------------------------------------------------------------
    # 4️⃣  Post‑process results – identical to your original script
    # --------------------------------------------------------------
    aggregated = defaultdict(lambda: defaultdict(list))
    for cfg_path, out in results.items():
        # extract optimizer name & epoch count from the temporary config filename
        # format: temp_sharpness_<db>_<opt>_<epochs>epochs_runXXXXX.yaml
        fname = pathlib.Path(cfg_path).stem
        parts = fname.split("_")
        opt_name = parts[2]
        epochs = int(parts[3].replace("epochs", ""))
        if out["sharpness"] is not None and out["accuracy"] is not None:
            aggregated[opt_name][epochs].append((out["sharpness"], out["accuracy"]))

    # Save JSON
    with open("sharpness_results.json", "w") as f:
        json.dump(aggregated, f, indent=4)

    # Plotting (exactly the same code you already had)
    generate_plots(aggregated)


# ----------------------------------------------------------------------
# Plotting helpers – unchanged from your original file (just accept the new dict shape)
# ----------------------------------------------------------------------
def generate_plots(results):
    """Two side‑by‑side plots: Sharpness vs Epochs & Accuracy vs Epochs."""
    plt.figure(figsize=(12, 5))

    # ---- Sharpness -------------------------------------------------
    plt.subplot(1, 2, 1)
    for opt, epoch_data in results.items():
        epochs = sorted(epoch_data.keys())
        means = [np.mean([s for s, _ in epoch_data[e]]) for e in epochs]
        stds = [np.std([s for s, _ in epoch_data[e]]) for e in epochs]
        plt.errorbar(epochs, means, yerr=stds, marker="o", capsize=5, label=opt)
    plt.xlabel("Epochs")
    plt.ylabel("Sharpness")
    plt.title("Sharpness vs Training Epochs")
    plt.legend()
    plt.grid(True)

    # ---- Accuracy --------------------------------------------------
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
    final_epoch = max(EPOCHS_RANGE)
    
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

# import yaml
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import time
# import random
# import torch
# import torch.multiprocessing as mp
# from collections import defaultdict
# from tqdm import tqdm
# from functools import partial
# import tempfile

# # Configuration
# database = "mnist"
# optimizers = ["sgd", "sgdm"]
# # optimizers = ["sgd", "sgdm", "adam", "easgd"]
# # optimizers = ["sgd", "sgdm", "adam", "entropysgd", "easgd"]

# # Produce logarithmically-spaced epochs array
# a1 = 10.**np.arange(1, 3)
# a2 = np.arange(1, 10, 2)
# epochs_range = np.outer(a1, a2).astype(np.int64).flatten().tolist()
# num_runs = 3  # Number of runs per configuration

# print(f"Your experiment will compare optimizers {optimizers} during task {database}")
# print(f"Your experiment will run each optimizer for {epochs_range} epochs with {num_runs} runs each")
# input("Are you ok with these settings? Press Enter to continue...")

# # Shared data storage
# shared_data = {}
# shared_model_class = None
# results = defaultdict(lambda: defaultdict(list))

# def init_worker(database_name, model_name, device_name):
#     """Initialize a worker process with shared data"""
#     import torch
#     from models import get_model
#     from data.loaders import get_data_loaders
    
#     # Set device
#     device = torch.device(device_name)
    
#     # Load data (once per process)
#     train_loader, val_loader, test_loader = get_data_loaders(
#         database_name,
#         batch_size=64  # You might want to make this configurable
#     )
    
#     # Get model class (not instance)
#     model_class = get_model(model_name)
    
#     # Store in process-global variables
#     global process_data, process_model_class, process_device
#     process_data = {
#         'train_loader': train_loader,
#         'val_loader': val_loader,
#         'test_loader': test_loader
#     }
#     process_model_class = model_class
#     process_device = device
    
#     print(f"Worker initialized with {database_name} and device {device_name}")

# def run_experiment(optimizer_name, epochs, run_id, pbar=None):
#     """Run a single experiment with modified epochs"""
#     # Determine config path
#     config_path = f"configs/{database}_{optimizer_name}.yaml"
    
#     # Load the base config
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)

#     # Set seeds for reproducibility
#     seed = run_id  # or use a fixed seed per run
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
    
#     # Update progress bar description
#     if pbar:
#         pbar.set_description(f"Running {optimizer_name}-{epochs}epochs-run{run_id}")
    
#     # Create a unique experiment name
#     # optimizer_name = os.path.basename(config_path).replace(f"{database}_", '').replace('.yaml', '')
#     experiment_name = f"sharpness_{database}_{optimizer_name}_{epochs}epochs_run{run_id}"
    
#     # Update the config
#     config['training']['epochs'] = epochs
#     config['logging']['log_dir'] = f"./logs/{experiment_name}"
    
#     # Save the modified config
#     temp_config_path = f"configs/temp_{experiment_name}.yaml"
#     with open(temp_config_path, 'w') as f:
#         yaml.dump(config, f, default_flow_style=False)
    
#     # Run the experiment
#     print(f"Running {experiment_name}...")
#     try:
#         print("trying subprocess...")
#         result = subprocess.run([
#             'python', 'main.py', '--silent', '--config', temp_config_path
#         ], capture_output=True, text=True, timeout=1800)  # 30-minute timeout

#         print(f"Subprocess completed with return code: {result.returncode}")

#         if result.returncode != 0:
#             print(f"Error running {experiment_name}: {result.stderr}")
#             # Also print stdout to see what happened
#             print(f"STDOUT: {result.stdout}")
#             return None, None
        
#         # Extract results
#         report_path = os.path.join(config['logging']['log_dir'], 'generalization_report.json')
#         if os.path.exists(report_path):
#             with open(report_path, 'r') as f:
#                 report = json.load(f)
#             return report.get('final_sharpness'), report.get('test_accuracy')
#         else:
#             print(f"Report not found for {experiment_name}")
#             # Check if the log directory exists and what's in it
#             log_dir = config['logging']['log_dir']
#             if os.path.exists(log_dir):
#                 print(f"Contents of {log_dir}: {os.listdir(log_dir)}")
#             return None, None
            
#     except subprocess.TimeoutExpired:
#         print(f"Experiment {experiment_name} timed out after 30 minutes")
#         return None, None
#     except Exception as e:
#         print(f"Error running {experiment_name}: {e}")
#         # Log the error to a file for later analysis
#         with open("failed_experiments.log", "a") as log_file:
#             log_file.write(f"{experiment_name}: {str(e)}\n")
#         return None, None
#     finally:
#         # Clean up temporary config
#         if os.path.exists(temp_config_path):
#             os.remove(temp_config_path)
#         # Update progress bar
#         if pbar:
#             pbar.update(1)

# def main():
#     # Calculate total number of experiments
#     total_experiments = len(optimizers) * len(epochs_range) * num_runs
    
#     # Create progress bar
#     with tqdm(total=total_experiments, desc="Overall Progress", unit="exp") as pbar:
#         # Run all experiments
#         for optimizer in optimizers:
#             if not os.path.exists(f"configs/{database}_{optimizer}.yaml"):
#                 print(f"Config file not found: configs/{database}_{optimizer}.yaml")
#                 # Update progress bar for skipped experiments
#                 pbar.update(len(epochs_range) * num_runs)
#                 continue
                
#             for epochs in epochs_range:
#                 for run in range(1, num_runs + 1):
#                     sharpness, accuracy = run_experiment(optimizer, epochs, run, pbar)
                    
#                     if sharpness is not None and accuracy is not None:
#                         results[optimizer][epochs].append((sharpness, accuracy))
                    
#                     # Add a small delay between runs
#                     time.sleep(2)
    
#     # Save results
#     with open('sharpness_results.json', 'w') as f:
#         # Convert to serializable format
#         serializable_results = {}
#         for optimizer, epoch_data in results.items():
#             serializable_results[optimizer] = {}
#             for epoch, values in epoch_data.items():
#                 serializable_results[optimizer][epoch] = [
#                     (float(sharpness), float(accuracy)) for sharpness, accuracy in values
#                 ]
#         json.dump(serializable_results, f, indent=4)
    
#     # Generate plots
#     generate_plots(results)

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