import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler

# First, check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class BaseModelOutputClass():
    def __init__(self):
        pass

    @staticmethod
    def model_output(data, model, *args, **kwargs):
        '''
        The model output function in TRAK paper
        :param data: data for model input, not restricted for the data type
        :param model: model to be traced
        '''
        pass

    @staticmethod
    def loss_grad_to_out(data, model, *args, **kwargs):
        '''
        The variable Q in TRAK paper
        :param data: data for model input, not restricted for the data type
        :param model: model to be traced
        '''
        pass

class MNISTModelOutput(BaseModelOutputClass):
    def __init__(self):
        super().__init__(self)

    @staticmethod
    def model_output(data, model, *args, **kwargs):
        image, label = data
        image, label = image.to(device), label.to(device)
        raw_logit = model(image)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        logp = -loss_fn(raw_logit, label)

        return logp - torch.log(1 - torch.exp(logp))

    @staticmethod
    def get_out_to_loss_grad(data, model, *args, **kwargs):
        image, label = data
        image, label = image.to(device), label.to(device)
        raw_logit = model(image)

        loss_fn = nn.CrossEntropyLoss(reduction='none')
        p = torch.exp(-loss_fn(raw_logit, label))

        return (1-p).clone().detach().unsqueeze(-1)

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10, num_layers=2):
        super(MLP, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.layers.append(torch.nn.Linear(hidden_size, output_size))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def train_with_seed(self, train_loader, epochs=30, seed=0, verbose=True):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        print("Training complete")

    def test(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        # No gradient is needed for evaluation
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                # Get the predicted class from the maximum value in the output-list of class scores
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    def get_model_output(self, test_loader):
        self.eval()
        model_output_list = []
        for _, data in enumerate(test_loader):
            model_output = MNISTModelOutput.model_output(data, self)
            model_output_list.append(model_output)
        model_output_tensor = torch.stack(model_output_list, dim=0)
        return model_output_tensor

class SubsetSamper(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def data_generation(train_size, test_size, subset_remove, mode='train'):
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # remove the corresponding index from the subset_remove
    sampler_train = SubsetSamper([i for i in range(train_size) if i not in subset_remove])

    # portion_index_test = np.random.choice([i for i in range(500)], size=500, replace=False, p=None)
    sampler_test = SubsetSamper([i for i in range(test_size)])

    if mode == 'train':
        train_batch_size, test_batch_size = 64, 64
    elif mode == 'eval':
        train_batch_size, test_batch_size = 64, 1
    elif mode == 'TRAK':
        train_batch_size, test_batch_size = 1, 1

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler_train)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, sampler=sampler_test)
    return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000, help="train dataset size")
    parser.add_argument("--test_size", type=int, default=500, help="test dataset size")
    parser.add_argument("--ensemble", type=int, default=5, help="ensemble number")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, test_loader = data_generation(args.train_size, args.test_size, [], mode='train')

    for ensemble_idx in range(args.ensemble):
        # Initialize the model, loss function, and optimizer
        model = MLP().to(device)
        model.train_with_seed(train_loader, epochs=30, seed=ensemble_idx)

        torch.save(model.state_dict(), f"./checkpoint/seed_{args.seed}_{ensemble_idx}.pt")

        model.test(test_loader)
