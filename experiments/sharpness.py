# sharpness.py
import yaml, json, os, time, random, subprocess, pathlib, numpy as np, matplotlib.pyplot as plt
import torch.multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict
from utils.shared_resources import init_shared_resources 
from main import run_experiment # entry point

# ----------------------------------------------------------------------
# Static configuration
# ----------------------------------------------------------------------
DATABASE = "mnist"
OPTIMIZERS = ["sgd", "sgdm"]
# OPTIMIZERS = ["sgd", "sgdm", "adam", "entropysgd", "easgd"]                     
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