# MISS

This is the official implementation of [Most Influential Subset Selection: Challenges, Promises, and Beyond](https://arxiv.org/abs/2409.18153).

## Setup Guide

In order to use this framework, you need to have a working installation of Python 3.8 or newer. The only uncommon package we're using is [pyDVL](https://pydvl.org/devel/). Please refer to the official guide from their website and correctly install it.

## Quick Start

Make sure you have followed the [Setup Guide](#setup-guide) before running the code.

### Linear Regression

The [linear_regression](linear_regression) directory consists of the key MISS algorithm (`LAGS.py`) and the Python notebooks for both the real-world experiment and synthetic data experiment. To obtain the result, simply run the notebooks.

### Logistic Regression

The [logistic_regression](logistic_regression) directory consists of the key MISS algorithm (`IF.py`) and the Python notebook for both the real-world experiment and synthetic data experiment. To obtain the result, simply run the notebooks.

### Multi-Layer Perceptron

The [MLP](MLP) directory mainly consists of the key MISS algorithm (`IF.py`), and a wrapper of the entire experiment (`MISS.py`) to obtain the result, with a python notebook for the evaluation (`evaluation_MNIST.ipynb`). We divide the workflow in several steps since this experiment is a bit time-consuming. We now detail the whole workflow.

>Before running the script, you will need to manually create the following directories: `./MLP/checkpoint`, `./MLP/checkpoint/adaptive_tmp`, `./MLP/results/Eval`, and `./MLP/results/IF`.

1. Train a number of models specified by `--ensemble`, and save them to `./MLP/checkpoint`.

	```bash
	python model_train.py --seed 0 --train_size 5000 --test_size 500 --ensemble 5
	```

	Note that the training set and the test set are constructed deterministically: in the above example, it'll take the first 5000 training samples and 500 test samples.

	>The test dataset here is only used to show the accuracy of the model; we do not use it for selecting the model (e.g., cross-validation). In other words, it won't affect the next step in any way.
2. Solve the MISS and save the result to `./MLP/results/IF`. For the naive greedy:

	```bash
	python MISS.py --seed 0 --train_size 5000 --test_range 0:49 --test_start_idx 0 --ensemble 5 --k 50
	```

	For the (stepped) adaptive greedy:

	```bash
	python MISS.py --seed 0 --train_size 5000 --test_range 0:49 --test_start_idx 0 --ensemble 5 --k 50 --adaptive --warm_start --step 5
	```

	Several notes on the flag:
	- `seed`: The seed used for the previous (step 1) experiment.
		>Note that step is deterministic (the training involved in this step is always controlled by some fixed seeds to avoid confusion).
	- `adaptive`: If specified, then the adaptive greedy will be used.
	- `warm_start` and `step`: These two flags only take effect when `adaptive` is specified.
	- `test_range`: Construct the test dataset with an index between the specified range in the format of `start:end` (inclusive).
		>This allows batched processing due to insufficient memory: initialization takes around 40 GB CUDA memory already, and after processing each test point the memory allocation increased by a non-negligible amount, which suffices to cause a CUDA out of memory error.
3. Run `evaluation_MNIST.ipynb` to evaluate the performance and generate plots. The evaluation result will be saved to `./MLP/results/Eval` if `load_eval` is set to `False` (you will need to do this at the first time).
	>The evaluation script will aggregate all batches in the second step together.

#### Examples

A sample script for the first two steps:

```bash
# Step 1
python3 model_train.py --seed 0 --train_size 5000 --test_size 500 --ensemble 5

# Step 2
## Greedy
python3 MISS.py --seed 0 --train_size 5000 --test_range 0:49 --ensemble 5 --k 50

## Adaptive Greedy
python3 MISS.py --seed 0 --train_size 5000 --test_range 0:24 --ensemble 5 --k 50 --adaptive --warm_start --step 5
python3 MISS.py --seed 0 --train_size 5000 --test_range 25:49 --ensemble 5 --k 50 --adaptive --warm_start --step 5
```

## Citations

If you find this repository valuable, please give it a star! Got any questions or feedback? Feel free to open an issue. Using this in your work? Please reference us using the provided citation:
