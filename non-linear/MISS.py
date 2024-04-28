from TRAK.MISS_trak import MISS_TRAK
from IF.MISS_IF import MISS_IF
from model_train import MLP, SubsetSamper, MNISTModelOutput
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

# First, check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=int, default=5, help="ensemble number")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--k", type=int, default=10, help="size of the most influential subset")
    args = parser.parse_args()

    if args.ensemble > 200:
        exit(0)

    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    sampler_train = SubsetSamper([i for i in range(5000)])
    sampler_test = SubsetSamper([i for i in range(500)])

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler_train)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=sampler_test)

    checkpoint_files = [f"./checkpoint/seed_{args.seed}_ensemble_{i}.pt" for i in range(args.ensemble)]

    trak = MISS_TRAK(model=MLP().to(device),
                     model_checkpoints=checkpoint_files,
                     train_loader=train_loader,
                     test_loader=test_loader,
                     model_output_class=MNISTModelOutput,
                     device=device)

    MISS = trak.most_k(args.k)
    torch.save(MISS, f"./MISS/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}.pt")

    MISS_adaptive = trak.adaptive_most_k(args.k)
    torch.save(MISS, f"./MISS/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_adaptive.pt")

    # Retrain the model without the most influential samples for every test point