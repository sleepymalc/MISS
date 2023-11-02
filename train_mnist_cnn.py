import argparse
import copy
import logging
import numpy as np
import os
import os.path as osp
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, Dataset

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer (28x28x1 -> 26x26x10)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        # Second convolutional layer (13x13x10 -> 11x11x20)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # Global average pooling layer (11x11x20 -> 20)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # Output layer (20 -> 2)
        self.output = nn.Linear(20, 2)

    def forward(self, x):
        # Apply first convolutional layer and pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Apply second convolutional layer and pooling
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # Apply global average pooling
        x = self.global_avg_pool(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        # Apply output layer
        x = F.softmax(self.output(x), dim=1)
        return x

class Model(nn.Module):
    def __init__(self, input_size, n_hidden, num_classes):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, n_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, num_classes)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.X[index], 0 if self.y[index]==1 else 1

class CustomDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        image, label = self.original_dataset[index]
        label = 0 if label == args.label_1 else 1
        return image, label

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--n-epochs', '-ep', type=int, default=5)
    parser.add_argument('--n-hidden', type=int, default=2000)
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--label-1', type=int, default=0)
    parser.add_argument('--label-2', type=int, default=1)
    parser.add_argument('--begin-idx', '-bid', type=int, default=0)
    parser.add_argument('--end-idx', '-eid', type=int, default=1000000)
    return parser.parse_args()

def reset_all_randomness(seed):
    # 0. controlling randomness
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # For all GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

args = parse_args()
reset_all_randomness(args.seed)

# 1. Loading the MNIST Dataset with basic transformation
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 2. Filtering for labels 0 and 1
n_train = args.n_train
n_val = 100
n_test = 100

indices_1 = [i for i, (img, label) in enumerate(full_dataset) if label == args.label_1]
indices_2 = [i for i, (img, label) in enumerate(full_dataset) if label == args.label_2]
train_indices = np.random.choice(indices_1, size=n_train, replace=False).tolist() \
    + np.random.choice(indices_2, size=n_train, replace=False).tolist()
val_indices = np.random.choice(list(set(indices_1) - set(train_indices)), size=n_val, replace=False).tolist() \
    + np.random.choice(list(set(indices_2) - set(train_indices)), size=n_val, replace=False).tolist()
test_indices = np.random.choice(list(set(indices_1) - set(train_indices) - set(val_indices)), size=n_test, replace=False).tolist() \
    + np.random.choice(list(set(indices_2) - set(train_indices) - set(val_indices)), size=n_test, replace=False).tolist()

train_dataset = CustomDataset(Subset(full_dataset, train_indices))
val_dataset = CustomDataset(Subset(full_dataset, val_indices))
test_dataset = CustomDataset(Subset(full_dataset, test_indices))

# 4. Creating data loaders
batch_size = args.bs  # Adjust if needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


folder_sub = osp.join('results', f'train-cnn-std_lr-{args.lr}_epoch-{args.n_epochs}_bs-{args.bs}_seed-{args.seed}_ntrain-{args.n_train}_{args.label_1}_{args.label_2}')

if not osp.exists(folder_sub):
    os.makedirs(folder_sub)

torch.save({
    'train_dataset': train_dataset,
    'val_dataset': val_dataset,
    'test_dataset': test_dataset,
}, osp.join(folder_sub, 'dataset.pt'))

task_name = 'train'
setup_logger(task_name, os.path.join(folder_sub, f'train_cont.log'))
logger = logging.getLogger(task_name)

logger.info(f'args = {args}')

class Solve:
    def __init__(self, train_dataset, val_dataset, test_dataset) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.criterion = nn.CrossEntropyLoss()

    def reset_model_and_data(self, del_indices=[]):
        logger.info(f'state reset! remove sample {del_indices}')
        reset_all_randomness(args.seed)
        self.model = SimpleCNN().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=0.0001)

        # 1. Determine the indices that you want to keep.
        all_indices = set(range(len(self.train_dataset)))
        keep_indices = list(all_indices - set(del_indices))

        # 2. Use Subset to create a new dataset with only the kept indices.
        new_train_dataset = Subset(self.train_dataset, keep_indices)

        # 3. Create a DataLoader for the new dataset.
        batch_size = 64  # Adjust if needed
        self.train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True)

        # 4. Create folder for storing results
        self.folder_name = osp.join(folder_sub, f'remove-{del_indices}')
        if not osp.exists(self.folder_name):
            os.makedirs(self.folder_name)

        self.best_val_loss = np.inf
        self.best_val_acc = -1
        self.best_model = None
        self.best_epoch = -1

    def test(self, data_loader):
        self.model.eval()

        eval_losses = []
        output_list = []
        correct_list = []

        correct = 0
        correct_class = [0, 0]
        total = 0   
        with torch.no_grad():
            for X, y in data_loader:
                X = X.cuda()
                y = y.cuda()

                output = self.model(X)
                loss = self.criterion(output, y)
                _, predicted = torch.max(output.data, 1)

                output_list.append(output)
                eval_losses.append(loss.item())
                correct_list.append(predicted.eq(y.data).cpu())

                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                correct_class[0] += torch.logical_and(predicted==y, y==0).sum().item()
                correct_class[1] += torch.logical_and(predicted==y, y==1).sum().item()

        return correct, correct_class, total, np.mean(eval_losses), torch.vstack(output_list), torch.cat(correct_list)


    def train(self):
        t = time.time()

        for epoch in (pbar := tqdm(range(args.n_epochs))):
            # 1. training
            self.model.train()
            for i, (X, y) in enumerate(self.train_loader):
                X = X.cuda()
                y = y.cuda()

                self.optimizer.zero_grad()

                output = self.model(X)

                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                
            # 2. validation
            val_correct, val_correct_class, val_total, val_loss, _, _ = self.test(self.val_loader)
            pbar.set_description(f"val_loss = {val_loss : .4f}, val_acc = {val_correct / val_total : .4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_correct / val_total
                self.best_model = copy.deepcopy(self.model)
                self.best_epoch = epoch

        logger.info(f'training finishes using {time.time() - t} seconds!')
        self.model = self.best_model
        logger.info(f'best epoch = {self.best_epoch}, best val loss = {self.best_val_loss}')

    def final_test(self):
        train_correct, train_correct_class, train_total, train_loss, output, correct = self.test(self.train_loader)
        logger.info(f'train_correct = {train_correct}, train_correct_class = {train_correct_class}, train_total = {train_total}, train_loss = {train_loss}')

        test_correct, test_correct_class, test_total, test_loss, output, correct = self.test(self.test_loader)
        logger.info(f'test_correct = {test_correct}, test_correct_class = {test_correct_class}, test_total = {test_total}, test_loss = {test_loss}')
        torch.save({
                'output': output,
                'correct': correct,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_correct/train_total, 
                'test_acc': test_correct/test_total,
            }, osp.join(self.folder_name, 'result.pt')
        )

solve = Solve(train_dataset, val_dataset, test_dataset)

# full set
# solve.reset_model_and_data([])
# solve.train()
# solve.final_test()
# _, _, _, _, output, _ = solve.test(
#     DataLoader(solve.train_dataset, batch_size=batch_size, shuffle=False)
# )
# torch.save(output, osp.join(solve.folder_name, 'margin.pt'))

# leave one out sets
for i in tqdm(range(args.begin_idx, min(args.end_idx, 2*args.n_train))):
    solve.reset_model_and_data([i])
    solve.train()
    solve.final_test()