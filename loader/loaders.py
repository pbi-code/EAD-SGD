import torch
from torchvision import datasets, transforms

def get_dataloader(dataset_name, data_path, batch_size, num_workers):
    """Create data loaders for the specified dataset"""
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        
    elif dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_set = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

