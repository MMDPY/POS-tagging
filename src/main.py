# ============================
# CMPUT 501 - ASSIGNMENT 4
# MOHAMMAD KARIMI & ALI ZAMANI
# ============================

# PREREQUISITES
import argparse
import sys
import warnings
import nltk
import numpy as np
from sklearn.model_selection import KFold
from nltk.tag.brill import Word, Pos, fntbl37, nltkdemo18, brill24, nltkdemo18plus
from nltk.tbl import Template
from nltk.tag import RegexpTagger, BrillTaggerTrainer
from nltk.probability import (
    SimpleGoodTuringProbDist,
    LaplaceProbDist,
    MLEProbDist,
    ELEProbDist,
    WittenBellProbDist,
)

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def load_tagged_sentences(path):
    """ Loads sentences from train and test text files
    Parameters:
        path (string): the path to the train/test file
    Returns:
        tagged_sentences (list) : a list of tuples containing word and POS tag pairs
    """
    tagged_sentences = []
    with open(path, 'r', encoding='utf8') as f:
        s = []
        for line in f:
            if line == '\n':
                tagged_sentences.append(s)
                s = []
            else:
                s.append((line.split()[0].strip(), line.split()[1].strip()))
    return tagged_sentences


def kfold_data(path, number_of_splits):
    """ Splits the data into train and development sets
        Parameters:
            path (string): the path to the train file
            number_of_splits (int): the number of splits of the K_fold object
        Returns:
            data_kfold (list) : a list containing dictionaries including the train and development folds
        """
    data = load_tagged_sentences(path)
    kf = KFold(n_splits=number_of_splits, random_state=27, shuffle=True)
    data_kfold = []
    for train_index, dev_index in kf.split(data):
        data_train = [data[i] for i in train_index]
        data_dev = [data[i] for i in dev_index]
        data_kfold.append({'data_train': data_train, 'data_dev': data_dev})
    return data_kfold


def cal_acc_tagger(tagger_type, list_data_kfold=None, estimator=None, template=None, max_rule=None):
    """ Calculates the average accuracy of the tagger using the given tagger and k-folded data
        Parameters:
            tagger_type (string): either hmm or brill
            list_data_kfold (list): list of k-folded data containing train and development sets
            estimator (object): a estimator object of the nltk.probability class
            template (object): predefined rules (template objects) from the nltk.brill class
            max_rule (int): maximum allowed number of rules
        Returns:
            acc (float) : average accuracy of the tuning process
        """

    accs = []
    for data_kfold in list_data_kfold:
        data_train = data_kfold['data_train']
        data_dev = data_kfold['data_dev']
        if tagger_type == "hmm":
            tagger_hmm = train_hmm_tagger(data_train, estimator)
            acc_hmm, _ = test_tagger(tagger_hmm, data_dev)
            accs.append(acc_hmm)
        elif tagger_type == "brill":
            tagger_brill = train_brill_tagger(data_train, template, max_rule)
            acc_brill, _ = test_tagger(tagger_brill, data_dev)
            accs.append(acc_brill)

    acc = np.mean(accs)
    return acc


def train_hmm_tagger(data_train, estimator):
    """ Trains the hmm tagger given the input data and the estimator
        Parameters:
            data_train (list): a list containing the train data
            estimator (object): a estimator object of the nltk.probability class
        Returns:
            tagger (object) : the tagger object trained on the given data and corresponding estimator
        """

    trainer = nltk.tag.hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(data_train, estimator=estimator)
    return tagger


def train_brill_tagger(data_train, template, max_rule):
    """ Trains the brill tagger given the input data, template, and max rule
        Parameters:
            data_train (list): a list containing the train data
            template (object): predefined rules from the nltk.brill class
            max_rule (int): maximum allowed number of rules
        Returns:
            tagger (object) : the tagger object trained on the given data and corresponding template and max rule
        """
    # the baseline (backoff) of the nltk.brill class
    baseline = RegexpTagger([
                      (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
                      (r'(The|the|A|a|An|an)$', 'AT'),   # articles
                      (r'.*able$', 'JJ'),                # adjectives
                      (r'.*ness$', 'NN'),                # nouns formed from adjectives
                      (r'.*ly$', 'RB'),                  # adverbs
                      (r'.*s$', 'NNS'),                  # plural nouns
                      (r'.*ing$', 'VBG'),                # gerunds
                      (r'.*ed$', 'VBD'),                 # past tense verbs
                      (r'.*', 'NN')                      # nouns
    ])

    bt = BrillTaggerTrainer(baseline, template, trace=3)
    Template._cleartemplates()
    tagger = bt.train(data_train, max_rules=max_rule)
    return tagger


def test_tagger(tagger, data_dev):
    """ Tests the performance of the tagger given the input data
        Parameters:
            tagger (object): an object of the nltk.brill or nltk.hmm class
            data_dev (list): dev/test split data
        Returns:
            acc (float): accuracy of the prediction on the test/dev set w.r.t the ground truth
            predicted_tag_list (list): the predicted tag list
    """
    predicted_tag_list = []
    for item_dev in data_dev:
        sentence_dev = [sent_tag[0] for sent_tag in item_dev]
        predicted_tag = tagger.tag(sentence_dev)
        predicted_tag_list.append(predicted_tag)
    acc = cal_accuracy(data_dev, predicted_tag_list)
    return acc, predicted_tag_list


def cal_accuracy(ground_truth, prediction):
    """ Calculates the accuracy of the prediction based on the ground truth and model's prediction
        Parameters:
            ground_truth (list): the list containing the words and their associated POS tags
            prediction (list): the list containing the words and their corresponding predictions
        Returns:
            acc (float): accuracy of the prediction on the test/dev set w.r.t the ground truth
    """

    total = 0
    correct = 0
    for i in range(len(ground_truth)):
        actual_sent = ground_truth[i]
        predicted_sent = prediction[i]
        for j in range(len(actual_sent)):
            actual_tag = actual_sent[j][1]
            predicted_tag = predicted_sent[j][1]
            if actual_tag == predicted_tag:
                correct += 1
            total += 1
    acc = correct / total
    return acc


def write_to_output(output_path, tagged_sentences):
    """ Writes the prediction result (POS tags) to the desired text file
        Parameters:
            output_path (string): the path to the output file
            tagged_sentences (list): the list containing the words and their corresponding predictions
        Returns:
            N/A
    """

    with open(output_path, 'w', encoding='utf8') as output_file:
        for tagged_sentence in tagged_sentences:
            for word_tag in tagged_sentence:
                word = word_tag[0]
                tag = word_tag[1]
                output_file.write(word + ' ' + tag + '\n')
            output_file.write('\n')


def main():
    # initializing the arg parser
    parser = argparse.ArgumentParser(description='Part of Speech tagging using HMM | Brill tagger')
    # argument for tagger
    parser.add_argument('--tagger', required=True, help='choice of POS tagger',
                        choices=['hmm', 'brill', 'trainhmm', 'trainbrill'])
    # argument for train path
    parser.add_argument('--train', required=True, help='path to the train file')
    # argument for test path
    parser.add_argument('--test', required=True, help='path to the test file')
    # argument for output path
    parser.add_argument('--output', required=True, help='path to the output file')
    # parsing the input args
    args = parser.parse_args()

    train_path = args.train
    test_path = args.test
    output_path = args.output
    tagger = args.tagger

    if tagger == 'hmm':
        data_test = load_tagged_sentences(test_path)
        data_train = load_tagged_sentences(train_path)
        tagger = train_hmm_tagger(data_train, estimator=WittenBellProbDist)
        acc, tagged_sentences = test_tagger(tagger, data_test)
        print("Fine-tuned HMM model accuracy on the test data: {}".format(acc*100))
        write_to_output(output_path, tagged_sentences)

    elif tagger == 'brill':
        data_test = load_tagged_sentences(test_path)
        data_train = load_tagged_sentences(train_path)
        tagger = train_brill_tagger(data_train,  template=fntbl37(), max_rule=100)
        acc, tagged_sentences = test_tagger(tagger, data_test)
        print("Fine-tuned BRILL model accuracy on the test data: {}".format(acc*100))
        write_to_output(output_path, tagged_sentences)

    elif tagger == 'trainhmm':
        estimators = [
            MLEProbDist,
            ELEProbDist,
            LaplaceProbDist,
            WittenBellProbDist,
            SimpleGoodTuringProbDist,
        ]
        performance_list = []
        for e in estimators:
            acc = cal_acc_tagger('hmm', kfold_data(train_path, 5), estimator=e)
            performance_list.append(acc*100)
        
        print('Brill tagger performance: {}'.format(performance_list))
        print('The best estimator is: {}'.format(estimators[np.argmax(performance_list)]))
        print('The best performance with the above-mentioned estimator is: {}'.format(max(performance_list)))

    elif tagger == 'trainbrill':
        templates = {
            'baseline': [Template(Pos([-1])), Template(Pos([-1]), Word([0]))],
            'brill24': brill24(),
            'nltkdemo18': nltkdemo18(),
            'nltkdemo18plus': nltkdemo18plus(),
            'fntbl37': fntbl37()
        }

        rule_counts = [10, 50, 100]
        performance_dict = {}
        for template_name, template in templates.items():
            # print('Template: {}'.format(template_name))
            for rule_count in rule_counts:
                # print('rule count: {}'.format(rule_count))
                acc = cal_acc_tagger('brill', kfold_data(train_path, 2),
                                     template=template, max_rule=rule_count)
                performance_dict[(template_name, rule_count)] = acc*100
        
        print('Brill tagger performance: {}'.format(performance_dict))
        best_performance = max(performance_dict.values())
        best_performance_idx = list(performance_dict.values()).index(best_performance)
        best_performance_keys = list(performance_dict.keys())
        best_template_rlue_count = best_performance_keys[best_performance_idx]
        print('The best template and rule count is: {}'.format(best_template_rlue_count))
        print('The best performance is: {}'.format(best_performance))


if __name__ == '__main__':
    main()
