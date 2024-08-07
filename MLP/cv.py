import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utlis.data import data_generation

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

    def train_with_seed(self, train_loader, epochs=30, seed=0, lr=0.01, momentum=0.9, verbose=True):
        torch.manual_seed(seed)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
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
        if verbose:
            print("Training complete")

    def test(self, test_loader, verbose=True):
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
        if verbose:
            print(f'Accuracy of the model on the test set: {accuracy:.2f}%')
        return accuracy

    def get_individual_output(self, test_loader):
        self.eval()
        model_output_list = []
        for _, data in enumerate(test_loader):
            model_output = MNISTModelOutput.model_output(data, self)
            model_output_list.append(model_output)
        model_output_tensor = torch.stack(model_output_list, dim=0)
        return model_output_tensor

    def get_individual_loss(self, test_loader):
        self.eval()
        loss_list = []
        for _, data in enumerate(test_loader):
            image, label = data
            image, label = image.to(device), label.to(device)
            raw_logit = self(image)

            loss_fn = nn.CrossEntropyLoss(reduction='none')
            logp = -loss_fn(raw_logit, label)
            loss_list.append(logp)
        loss_tensor = torch.cat(loss_list, dim=0)
        return loss_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000, help="train dataset size")
    parser.add_argument("--test_size", type=int, default=500, help="test dataset size")
    parser.add_argument("--fold", type=int, default=5, help="ensemble number")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    for lr in [0.01, 0.05, 0.1, 0.5]:
        for momentum in [0.9, 0.95]:
            for epochs in [30, 50]:
                for width in [64, 128]:
                    auc = 0
                    for i in range(args.fold):
                        train_loader, test_loader = data_generation(list(range(i * args.train_size, (i+1)*args.train_size)), list(range(args.test_size)), mode='train')

                        # Initialize the model, loss function, and optimizer
                        model = MLP(hidden_size=width).to(device)
                        model.train_with_seed(train_loader, epochs=epochs, seed=i, lr=lr, momentum=momentum, verbose=False)

                        auc += model.test(test_loader, verbose=False)
                    print(f"width: {width}\t| lr: {lr}\t| mom: {momentum}\t| epochs: {epochs}\t| auc: {auc / 5}")
