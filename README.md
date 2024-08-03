# MISS

## Linear Regression

## Logistic Regression

## Multi-Layer Perceptron

Before running the script, you will need to manually create the following directories: `./MLP/checkpoint`, `./MLP/checkpoint/adaptive_tmp`, `./MLP/results/Eval`, and `./MLP/results/IF`.

1. Train a number of models specified by `--ensemble`, and save them to `./MLP/checkpoint`.
	```bash
	python model_train.py --train_size 5000 --test_size 500 --ensemble 5 --seed 0
	```
	>Here the test dataset is used to show the final accuracy purely, nothing else (didn't use it for cross-validation, etc.). In other words, it won't affect the next step in any way.
2. Solve the MISS using both pure greedy and stepped adaptive greedy algorithm, and save the result to `./MLP/results/IF`.
	```bash
	python MISS.py --train_size 5000 --test_size 25 --test_start_idx 0 --ensemble 5 --seed 0 --k 50 --step 5 --warm_start
	```
	>Several notes:
	>1. You can use *warm start* training during the adaption by specifying `--warm_start` as above.
	>2. In the paper, we have `test_size=50`. However, due to memory constraints (initialization takes around 40 GB CUDA memory), we split the experiment into smaller batches, where each experiment only considers test data points with index between `test_start_idx` and `test_start_idx + test_size`.
	>3. Although not required, but the `seed` used in this step should be consistent as the first step to avoid any confusion.
3. Run `real_world.ipynb` to evaluate the performance and generate plots. The evaluation result will be saved to `./MLP/results/Eval` if `load_eval` is set to `False` (you will need to do this at the first time).

A sample script might look something like this:

```bash
python3 model_train.py --seed 0 --train_size 5000 --test_size 500 --ensemble 5
python3 MISS.py --seed 0 --train_size 5000 --test_size 25 --test_start_idx 0 --ensemble 5  --k 50 --warm_start --naive
python3 MISS.py --seed 0 --train_size 5000 --test_size 25 --test_start_idx 0 --ensemble 5  --k 50 --warm_start --adaptive --step 5
python3 MISS.py --seed 0 --train_size 5000 --test_size 25 --test_start_idx 25 --ensemble 5  --k 50 --warm_start --adaptive --step 5
```