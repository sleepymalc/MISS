# MISS

## Linear Regression

## Logistic Regression

## Multi-Layer Perceptron

> Before running the script, you will need to manually create the following directories: `./MLP/checkpoint`, `./MLP/checkpoint/adaptive_tmp`, `./MLP/results/Eval`, and `./MLP/results/IF`.

1. `python train_model.py --train_size 5000 --test_size 500 --ensemble 5 --seed 0`. This will train a number of models specified by `--ensemble`, and save them to `./MLP/checkpoint`.
2. `python MISS.py --train_size 5000 --test_size 500 --ensemble 5 --seed 0 --k 50 --step 5`. This runs the MISS computation, and save the result to `./MLP/results/IF`.
	> It's a 2d tensor, each row corresponds to the $k$-most influential subset selected by the algorithm (influence function) for a particular test point.
3. run `real_world.ipynb` to evaluation and generate plots. The evaluation result will be saved to `./MLP/results/Eval` if `load_eval` is set to `False` (you will need to do this at the first time).