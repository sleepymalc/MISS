from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler

class SubsetSamper(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def data_generation(train_indices, test_indices, mode='train'):
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if mode == 'train':
        train_batch_size, test_batch_size = 64, 64
    elif mode == 'eval':
        train_batch_size, test_batch_size = 64, 1
    elif mode == 'MISS':
        train_batch_size, test_batch_size = 1, 1

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=SubsetSamper(train_indices))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=SubsetSamper(test_indices))

    return train_loader, test_loader