# utils/shared_resources.py
import torch
from loader.loaders import get_dataloader

# ----------------------------------------------------------------------
# These objects are filled once by the master process (sharpness.py)
# and then handed to every worker via the Pool initializer.
# ----------------------------------------------------------------------
_shared = {
    "device": None,          # torch.device
    "train_loader": None,    # DataLoader
    "test_loader": None,     # DataLoader
    "model_factory": None,   # Callable that returns a fresh model instance
    "num_classes": None,     # int
}


def init_shared_resources(cfg_path: str):
    """
    Called once per worker (via Pool initializer).  It reads the yaml
    configuration, builds the device, the two data loaders and a *factory*
    that can create a brand new model instance on demand.
    """
    import yaml
    from models.lenet import LeNet
    from models.resnet import ResNet18

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ----- device ---------------------------------------------------------
    _shared["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- data loaders ---------------------------------------------------
    train_loader, test_loader = get_dataloader(
        cfg["dataset"],
        cfg["data_path"],
        cfg["batch_size"],
        cfg["num_workers"],
    )
    _shared["train_loader"] = train_loader
    _shared["test_loader"] = test_loader

    # ----- number of classes ---------------------------------------------
    if cfg["dataset"] == "CIFAR10":
        _shared["num_classes"] = 10
    elif cfg["dataset"] == "MNIST":
        _shared["num_classes"] = 10
    else:
        raise ValueError(f"Unsupported dataset {cfg['dataset']}")

    # ----- model factory --------------------------------------------------
    if cfg["model"] == "LeNet":
        _shared["model_factory"] = lambda: LeNet(num_classes=_shared["num_classes"])
    elif cfg["model"] == "ResNet18":
        _shared["model_factory"] = lambda: ResNet18(num_classes=_shared["num_classes"])
    else:
        raise ValueError(f"Unsupported model {cfg['model']}")