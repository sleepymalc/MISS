from IF import MISS_IF
from model_train import MLP, MNISTModelOutput
from utlis.data import data_generation
import torch
import argparse

# First, check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def parse_range(value):
    try:
        start, end = map(int, value.split(':'))
        if start < 0 or end < 0 or start > end:
            raise argparse.ArgumentTypeError("Invalid range: start should be less or equal than end and both should be non-negative")
        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid range format: expected 'start:end'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--train_size", type=int, default=5000, help="train dataset size")
    parser.add_argument("--test_range", type=parse_range, required=True, help="test dataset range. Format: 'start:end'")
    parser.add_argument("--ensemble", type=int, default=5, help="ensemble number")
    parser.add_argument("--k", type=int, default=50, help="size of the most influential subset")
    parser.add_argument("--adaptive", action='store_true', help="use adaptive greedy")
    parser.add_argument("--warm_start", action='store_true', help="enable warm start for adaptive greedy")
    parser.add_argument("--step", type=int, default=5, help="step size for adaptive greedy")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    test_start_idx, test_end_idx = args.test_range

    train_loader, test_loader = data_generation(list(range(args.train_size)), list(range(test_start_idx, test_end_idx + 1)), mode='MISS')

    checkpoint_files = [f"./checkpoint/seed_{args.seed}_ensemble_{i}.pt" for i in range(args.ensemble)]

    IF = MISS_IF(model=MLP().to(device),
                 model_checkpoints=checkpoint_files,
                 train_loader=train_loader,
                 test_loader=test_loader,
                 ensemble=args.ensemble,
                 model_output_class=MNISTModelOutput,
                 seed=args.seed,
                 device=device)

    if not args.adaptive: # MISS with naive greedy
        MIS = IF.most_k(args.k)
        torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_test_{test_start_idx}-{test_end_idx}.pt")
    else: # MISS with adaptive greedy
        MIS = IF.adaptive_most_k(args.k, warm_start=args.warm_start, step_size=args.step)
        if args.warm_start:
            torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_adaptive_step_{args.step}_warm-start_test_{test_start_idx}-{test_end_idx}.pt")
        else:
            torch.save(MIS, f"./results/IF/seed_{args.seed}_k_{args.k}_ensemble_{args.ensemble}_adaptive_step_{args.step}_test_{test_start_idx}-{test_end_idx}.pt")
