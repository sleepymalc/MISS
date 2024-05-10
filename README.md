# MISS

## Linear Regression

## Logistic Regression

## Multi-Layer Perceptron

1. `python train_model.py --train_size 5000 --test_size 500 --ensemble 5 --seed 0`. This will train a number of models specified by `--ensemble`, and save them to `./checkpoint`
2. `python ./TRAK/MISS.py --train_size 5000 --test_size 500 --ensemble 5 --seed 0 --k 5`. This runs the MISS computation using TRAK, and save the result to `./results/TRAK`.
	> It's a 2d tensor, each row corresponds to the $k$-most influential subset selected by the algorithm (either TRAk or TRAK adaptive) for a particular test point.
3. run `results.ipynb` to generate plots.