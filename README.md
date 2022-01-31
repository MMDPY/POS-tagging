# Intro to NLP - Assignment 4

## Team
|Student name| CCID |
|------------|------|
|student 1   |   karimiab   |
|student 2   |   azamani1   |

Please note that CCID is **different** from your student number.

## TODOs

In this file you **must**:
- [x] Fill out the team table above. 
- [x] Make sure you submitted the URL on eClass.
- [x] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment.
- [x] Provide clear installation and execution instructions that TAs must follow to execute your code.
- [x] List where and why you used 3rd-party libraries.
- [x] Delete the line that doesn't apply to you in the Acknowledgement section.

## Acknowledgement 
In accordance with the UofA Code of Student Behaviour, we acknowledge that  
(**delete the line that doesn't apply to you**)

- We did not consult any external resource for this assignment.
- We have listed all external resources we consulted for this assignment.

 Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.

## 3-rd Party Libraries
You do not need to list `nltk` and `pandas` here.

* `main.py L:[12]` used `[sklearn.model_selection]` for [importing the KFold class object].
* `main.py L:[90]` used `[numpy]` for [using numpy.mean to take average of the list].
* `main.py L:[246]` used `[numpy]` for [using numpy.argmax to find the index of the maximum accuracy].

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
