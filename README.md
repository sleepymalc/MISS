# MISS

## Linear Regression

## Logistic Regression

## Multi-Layer Perceptron

Before running the script, you will need to manually create the following directories: `./MLP/checkpoint`, `./MLP/checkpoint/adaptive_tmp`, `./MLP/results/Eval`, and `./MLP/results/IF`.

1. Train a number of models specified by `--ensemble`, and save them to `./MLP/checkpoint`.
	```bash
	python model_train.py --seed 0 --train_size 5000 --test_size 500 --ensemble 5
	```
	Note that the training set and the test set is constructed deterministically: in the above example, it'll take the first 5000 training samples and 500 test samples.

	>Here the test dataset is used to show the final accuracy purely, nothing else (didn't use it for cross-validation, etc.). In other words, it won't affect the next step in any way.
2. Solve the MISS and save the result to `./MLP/results/IF`. For the naive greedy:
	```bash
	python MISS.py --seed 0 --train_size 5000 --test_size 50 --test_start_idx 0 --ensemble 5 --k 50
	```
	For the (stepped) adaptive greedy:
	```bash
	python MISS.py --seed 0 --train_size 5000 --test_size 50 --test_start_idx 0 --ensemble 5 --k 50 --adaptive --warm_start --step 5
	```
	Several notes on the flag:
	- `seed`: The whole seed of the experiment.
		>Although not required, but the `seed` used in this step should be consistent as the first step to avoid any confusion.
	- `adaptive`: If specified, then the adaptive greedy will be used.
	- `warm_start` and `step`: These two flags only takes effect when `adaptive` is specified.
	- `test_start_idx`: Construct the test dataset with index between `test_start_idx` and `test_start_idx + test_size`.
		>We use `test_start_idx` due to insufficient memory: initialization takes around 40 GB CUDA memory already, and after processing each test point the memory allocation increased by a non-negligible amount, which suffices to cause a CUDA out of memory error.
3. Run `evaluation.ipynb` to evaluate the performance and generate plots. The evaluation result will be saved to `./MLP/results/Eval` if `load_eval` is set to `False` (you will need to do this at the first time).
	>The evaluation script will aggregates all batches in the second step together.

A sample script for the first two steps:

```bash
python3 model_train.py --seed 0 --train_size 5000 --test_size 500 --ensemble 5

python3 MISS.py --seed 0 --train_size 5000 --test_size 50 --test_start_idx 0 --ensemble 5 --k 50

python3 MISS.py --seed 0 --train_size 5000 --test_size 25 --test_start_idx 0 --ensemble 5 --k 50 --adaptive --warm_start --step 5
python3 MISS.py --seed 0 --train_size 5000 --test_size 25 --test_start_idx 25 --ensemble 5 --k 50 --adaptive --warm_start --step 5
```