# main.py
import argparse, os, sys, yaml, gc, torch, json
from loader import get_dataloader
from utils.logger import Logger
from training.train import Trainer
from training.EASGD_optimizer import EA_SGD
from training.EntropySGD import EntropySGD
from torch.optim import SGD, Adam
from utils.shared_resources import _shared

# Import models
from models.lenet import LeNet
from models.resnet import ResNet18

# ----------------------------------------------------------------------
# Helper: build optimizer from a config dict
# ----------------------------------------------------------------------
def get_optimizer(opt_cfg, model):
    name = opt_cfg["name"]
    if name == "EASGD":
        # k_sparsity_schedule may be missing – default handled inside optimizer
        return EA_SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            inner_lr=opt_cfg["inner_lr"],
            inner_steps=opt_cfg["inner_steps"],
            thermal_noise=opt_cfg.get("thermal_noise", 1e-3),
            projection_freq=opt_cfg["projection_freq"],
            k_sparsity=opt_cfg.get("k_sparsity", 0.1),
            gamma=opt_cfg["gamma"],
            momentum=opt_cfg.get("momentum", 0.0),
            weight_decay=opt_cfg["weight_decay"],
            kfac_update_freq=opt_cfg["kfac_update_freq"],
            k_sparsity_schedule=opt_cfg.get("k_sparsity_schedule"),
        )
    if name == "SGD":
        return SGD(
            model.parameters(),
            lr=opt_cfg["lr"],
            momentum=opt_cfg.get("momentum", 0.0),
            weight_decay=opt_cfg["weight_decay"],
        )
    if name == "Adam":
        return Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
            betas=opt_cfg.get("betas", (0.9, 0.999)),
        )
    if name == "EntropySGD":
        return EntropySGD(
            model.parameters(),
            config={
                "lr": opt_cfg["lr"],
                "momentum": opt_cfg.get("momentum", 0.0),
                "weight_decay": opt_cfg["weight_decay"],
                "nesterov": opt_cfg.get("nesterov", True),
                "L": opt_cfg["L"],
                "eps": opt_cfg["eps"],
                "g0": opt_cfg["g0"],
                "g1": opt_cfg["g1"],
            },
        )
    raise ValueError(f"Unsupported optimizer {name}")


# ----------------------------------------------------------------------
# Core experiment runner – this is what the Pool will call.
# ----------------------------------------------------------------------
def run_experiment(
    config_path,
    *,
    repeat=False, # flag for repeat call to main.run_experiment(), will trigger shared resources. Use this for sweeps
    silent=False, # suppress verbose output
    resume=None,  # resume from pause point
):
    """
    Execute a single training run.
    If ``repeat`` is True the function expects that the heavy objects
    (device, loaders, model factory) are already stored in
    ``utils.shared_resources._shared``.
    """
    # --------------------------------------------------------------------------
    # Load config (always needed – it contains hyper‑params, logging dir, etc.)
    # --------------------------------------------------------------------------
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------------------------------------------------------------------------------
    # Grab shared resources (repeated experiment) or create them locally (single run))
    # ---------------------------------------------------------------------------------
    if repeat:
        device = _shared["device"]
        train_loader = _shared["train_loader"]
        test_loader = _shared["test_loader"]
        model_factory = _shared["model_factory"]
        num_classes = _shared["num_classes"]
    else:
        # This branch is used when you invoke ``python main.py`` directly.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, test_loader = get_dataloader(
            cfg["dataset"],
            cfg["data_path"],
            cfg["batch_size"],
            cfg["num_workers"],
        )
        # Build a tiny factory that produces a fresh model each call.
        if cfg["model"] == "LeNet":
            model_factory = lambda: LeNet(num_classes=10)
        elif cfg["model"] == "ResNet18":
            model_factory = lambda: ResNet18(num_classes=10)
        else:
            raise ValueError(f"Unknown model {cfg['model']}")
        num_classes = 10  # both MNIST & CIFAR10 have 10 classes

    # --------------------------------------------------------------
    # Model & logger
    # --------------------------------------------------------------
    model = model_factory().to(device)

    log_dir = cfg["logging"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    logger = Logger(
        log_dir,
        use_tensorboard=cfg["logging"]["use_tensorboard"],
        num_classes=num_classes,
        optimizer=None,               # will be attached later
    )
    # Persist the exact config used for this run
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # --------------------------------------------------------------
    # Optimizer (may need a reference to the model for EASGD)
    # --------------------------------------------------------------
    optimizer = get_optimizer(cfg["optimizer"], model)
    if cfg["optimizer"]["name"] == "EASGD":
        optimizer.set_model(model)      # <-- EASGD needs the model for its inner loop

    # Attach optimizer to logger (so TensorBoard sees LR, etc.)
    logger.optimizer = optimizer

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------
    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        test_loader,
        device,
        logger,
        cfg,
    )

    # --------------------------------------------------------------
    # Optional checkpoint resume
    # --------------------------------------------------------------
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if trainer.scheduler is not None and "scheduler_state_dict" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if not silent:
            print(f"Resumed from checkpoint {resume}")

    # --------------------------------------------------------------
    # Train
    # --------------------------------------------------------------
    try:
        if not silent:
            print("=== Training started ===")
        trainer.train()
        if not silent:
            print("=== Training finished ===")
    except Exception as exc:
        print(f"Training crashed: {exc}")
        raise
    finally:
        # ----------------------------------------------------------
        # Clean‑up – identical to what you already had
        # ----------------------------------------------------------
        if hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "close"):
            train_loader.dataset.close()
        if hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "close"):
            test_loader.dataset.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.close()
        if not silent:
            print("Resources released, exiting.")

    # --------------------------------------------------------------
    # Return a tiny dict that sharpness.py can aggregate
    # --------------------------------------------------------------
    return {
        "sharpness": getattr(trainer, "final_sharpness", None),
        "accuracy": getattr(trainer, "final_accuracy", None),
    }


# ----------------------------------------------------------------------
# CLI entry point – unchanged behaviour, just forwards the flag.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run a single optimizer experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--silent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--repeat",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Used by sharpness.py – reuse shared device/loaders",
    )
    args = parser.parse_args()

    # The function returns a dict; we ignore it for the CLI case.
    run_experiment(
        args.config,
        repeat=args.repeat,
        silent=args.silent,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()