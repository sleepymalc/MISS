from IF import MISS_IF
from model_train import MLP, MNISTModelOutput
from utlis.data import data_generation
import torch
import argparse

# First, check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000, help="train dataset size")
    parser.add_argument("--test_size", type=int, default=50, help="test dataset size")
    parser.add_argument("--ensemble", type=int, default=5, help="ensemble number")
    parser.add_argument("--warm_start", action='store_true', help="enable warm start")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--k", type=int, default=50, help="size of the most influential subset")
    parser.add_argument("--step", type=int, default=5, help="step size for adaptive influence function")
    args = parser.parse_args()

    batch_size = 1

    torch.manual_seed(args.seed)

    train_loader, test_loader = data_generation(list(range(args.train_size)), list(range(args.test_size)), mode='MISS')

    checkpoint_files = [f"./checkpoint/seed_{args.seed}_ensemble_{i}.pt" for i in range(args.ensemble)]

    IF = MISS_IF(model=MLP().to(device),
                 model_checkpoints=checkpoint_files,
                 train_loader=train_loader,
                 test_loader=test_loader,
                 ensemble=args.ensemble,
                 model_output_class=MNISTModelOutput,
                 warm_start=args.warm_start,
                 device=device)

    # MISS with greedy
    MIS = IF.most_k(args.k)
    if args.warm_start:
        torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_w.pt")
    else:
        torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}.pt")

    # MISS with adaptive greedy
    for start_idx in range(0, args.test_size, batch_size):
        end_idx = min(start_idx + batch_size, args.test_size)

        train_loader, test_loader = data_generation(list(range(args.train_size)), list(range(start_idx, end_idx)), mode='MISS')

        IF = MISS_IF(model=MLP().to(device),
                 model_checkpoints=checkpoint_files,
                 train_loader=train_loader,
                 test_loader=test_loader,
                 ensemble=args.ensemble,
                 model_output_class=MNISTModelOutput,
                 warm_start=args.warm_start,
                 device=device)

        MIS[start_idx:end_idx, :] = IF.adaptive_most_k(args.k, step_size=args.step)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.warm_start:
        torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_adaptive_step_{args.step}_w.pt")
    else:
        torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_adaptive_step_{args.step}.pt")
