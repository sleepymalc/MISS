from TRAK.MISS_trak import MISS_TRAK
from IF.MISS_IF import MISS_IF
from model_train import MLP, MNISTModelOutput, data_generation
import torch
import argparse

# First, check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000, help="train dataset size")
    parser.add_argument("--test_size", type=int, default=500, help="test dataset size")
    parser.add_argument("--ensemble", type=int, default=5, help="ensemble number")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--k", type=int, default=10, help="size of the most influential subset")
    args = parser.parse_args()

    train_loader, test_loader = data_generation(args.train_size, args.test_size, [], mode='TRAK')

    checkpoint_files = [f"./checkpoint/seed_{args.seed}_ensemble_{i}.pt" for i in range(args.ensemble)]

    trak = MISS_TRAK(model=MLP().to(device),
                     model_checkpoints=checkpoint_files,
                     train_loader=train_loader,
                     test_loader=test_loader,
                     model_output_class=MNISTModelOutput,
                     proj_dim=1000,
                     device=device)

    # TRAK
    MISS = trak.most_k(args.k)
    torch.save(MISS, f"./TRAK/results/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}.pt")

    # adaptive TRAK
    MISS_adaptive = trak.adaptive_most_k(args.k)
    torch.save(MISS_adaptive, f"./TRAK/results/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_adaptive.pt")

    # Retrain the model without the most influential samples for every test point