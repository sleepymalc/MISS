# MISS

## Linear Regression

## Logistic Regression

## Multi-Layer Perceptron

Before running the script, you will need to manually create the following directories: `./MLP/checkpoint`, `./MLP/checkpoint/adaptive_tmp`, `./MLP/results/Eval`, and `./MLP/results/IF`.

1. Train a number of models specified by `--ensemble`, and save them to `./MLP/checkpoint`.
	```python
	python model_train.py --train_size 5000 --test_size 500 --ensemble 5 --seed 0
	```
2. Solve the MISS using both pure greedy and stepped adaptive greedy algorithm, and save the result to `./MLP/results/IF`.
	```python
	python MISS.py --train_size 5000 --test_size 50 --ensemble 5 --seed 0 --k 50 --step 5 --warm_start
	```
	> You can use *warm start* training during the adaption by specifying `--warm_start` as above.
3. Run `real_world.ipynb` to evaluate the performance and generate plots. The evaluation result will be saved to `./MLP/results/Eval` if `load_eval` is set to `False` (you will need to do this at the first time).