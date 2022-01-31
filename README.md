# Part-of-Speech (POS) Tagging using Hidden Markov Model (HMM) and Brill-tagger

## Execution
Example usage: use the following command in the current directory.

### hyperparameter tuning
For running the hyperparameter tuning on the hmm tagger:

`python3 src/main.py --tagger trainhmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`

For running the hyperparameter tuning on the brill tagger:

`python3 src/main.py --tagger trainbrill --train data/train.txt --test data/test.txt --output output/test_brill.txt`

### testing

For running the hmm tagger with the best found estimator:

`python3 src/main.py --tagger hmm --train data/train.txt --test data/test.txt --output output/test_hmm.txt`

For running the hmm tagger with the best found estimator on the ood data:

`python3 src/main.py --tagger hmm --train data/train.txt --test data/test_ood.txt --output output/test_ood_hmm.txt`

For running the brill tagger with the best found template and rule count:

`python3 src/main.py --tagger brill --train data/train.txt --test data/test.txt --output output/test_brill.txt`

For running the hmm tagger with the best found estimator on the ood data:

`python3 src/main.py --tagger brill --train data/train.txt --test data/test_ood.txt --output output/test_ood_brill.txt`



## Data

The assignment's training data can be found in [data/train.txt](data/train.txt), the in-domain test data can be found in [data/test.txt](data/test.txt), and the out-of-domain test data can be found in [data/test_ood.txt](data/test_ood.txt).
