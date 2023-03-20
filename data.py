import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_train_dataset(validation_split=0.1, transform=None):
    """
    Return dataset for training and validation

    Args:
        validation_split (float): Fraction of the training dataset to be used for validation
        transform (torchvision.transforms.Compose): Transformations to apply to the dataset

    Returns:
        train_dataset
        valid_dataset
    """
    train_transform = transform or get_default_transform()
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    num_train_samples = len(train_dataset)
    num_valid_samples = int(num_train_samples * validation_split)
    num_train_samples -= num_valid_samples

    train_dataset, valid_dataset = random_split(train_dataset, [num_train_samples, num_valid_samples])

    return train_dataset, valid_dataset

def get_test_dataset(transform=None):
    """
    Return dataset for test

    Args:
        transform (torchvision.transforms.Compose): Transformations to apply to the dataset

    Returns:
        test_dataset
    """
    test_transform = transform or get_default_transform()
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    return test_dataset

def get_dataloader(dataset, batch_size, num_workers=8, shuffle=True):
    """
    Return torch dataloader for dataset

    Args:
        dataset: Torch dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading (default: 0)
        shuffle: Shuffle the dataset (default: True)

    Returns:
        data_loader: Torch data loader
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return data_loader

def get_default_transform():
    """
    Return default transformations to apply to the dataset

    Returns:
        transform: torchvision.transforms.Compose object
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform

