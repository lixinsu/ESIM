"""
Preprocess some NLI dataset and word embeddings to be used by the ESIM model.
"""
# Aurelien Coet, 2018.

import os
import pickle
import string
import fnmatch
import json
import ipdb
import numpy as np
from collections import Counter


def read_data(filepath, lowercase=False, ignore_punctuation=False):
    """
    Read the premises, hypotheses and labels from a file in some NLI
    dataset and return them in a dictionary.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (or MultiNLI) dataset.
        lowercase: Boolean value indicating whether the words in the premises
            and hypotheses must be lowercased.
        ignore_punctuation: Boolean value indicating whether to ignore
            punctuation in the premises and hypotheses.

    Returns:
        A dictionary containing three lists, one for the premises, one for the
        hypotheses, and one for the labels in the input data.
    """
    with open(filepath, 'r') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove parentheses and punctuation from
        # strings.
        parentheses_table = str.maketrans({'(': None, ')': None})
        punct_table = str.maketrans({key: ' ' for key in string.punctuation})

        # Ignore the headers on the first line of the file.
        next(input_data)

        for line in input_data:
            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            # Remove '(' and ')' from the premises and hypotheses.
            premise = premise.translate(parentheses_table)
            hypothesis = hypothesis.translate(parentheses_table)

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append(premise.rstrip().split())
            hypotheses.append(hypothesis.rstrip().split())
            labels.append(line[0])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


def read_data_sogou(filepath, **kwargs):
    premises, hypotheses, labels = [], [], []
    with open(filepath) as fin:
        for line in fin:
            data = json.loads(line)
            premises.append(data['passage'])
            hypotheses.append(data['question'])
            labels.append(data['label'])
    return {"premises": premises,
            "hypotheses": hypotheses,
            "labels": labels}


def build_worddict(data, num_words=None):
    """
    Build a dictionary associating words from a set of premises and
    hypotheses to unique integer indices.
    Update: we put the most common word in hypotheses

    Args:
        data: A dictionary containing the premises and hypotheses for which
            a worddict must be built. The dictionary is assumed to have the
            same form as the dicts built by the 'read_data' function of this
            module.
        num_words: Integer indicating the maximum number of words to
            keep in the worddict. If specified, only the 'num_words' most
            frequent words will be kept. If set to None, all words are
            kept. Defaults to None.

    Returns:
        A dictionary associating words to integer indices.
    """
    words = []
    [words.extend(sentence) for sentence in data['hypotheses']]
    counts = Counter(words)
    if num_words is None:
        num_words = len(counts)

    worddict = {word[0]: i+4
                for i, word in enumerate(counts.most_common(num_words))}

    worddict["_PAD_"] = 0
    worddict["_OOV_"] = 1
    worddict["_BOS_"] = 2
    worddict["_EOS_"] = 3
    #open('tmp_question_word.pkl', 'w').write(json.dumps(worddict))

    # Special indices are used for padding, out-of-vocabulary words, and
    # beginning and end of sentence tokens.
    [words.extend(sentence) for sentence in data['premises']]
    counts = Counter(words)
    num_words = len(counts)
    for word, v in counts.most_common(num_words):
        if word not in worddict:
            worddict[word] = len(worddict)

    return worddict


def words_to_indices(sentence, worddict):
    """
    Transform the words in a sentence to integer indices.

    Args:
        sentence: A list of words that must be transformed to indices.
        worddict: A dictionary associating words to indices.

    Returns:
        A list of indices.
    """
    # Include the beggining of sentence token at the start of the sentence.
    indices = [worddict["_BOS_"]]
    for word in sentence:
        if word in worddict:
            index = worddict[word]
        else:
            # Words absent from 'worddict' are treated as a special
            # out-of-vocabulary word (OOV).
            index = worddict['_OOV_']
        indices.append(index)
    # Add the end of sentence token at the end of the sentence.
    indices.append(worddict["_EOS_"])

    return indices


def transform_to_indices(data, worddict, labeldict):
    """
    Transform the words in the premises and hypotheses of a dataset, as well
    as their associated labels, to integer indices.

    Args:
        data: A dictionary containing lists of premises, hypotheses
            and labels.
        worddict: A dictionary associating words to unique integer indices.
        labeldict: A dictionary associating labels to unique integer indices.

    Returns:
        A dictionary containing the transformed premises, hypotheses and
        labels.
    """
    transformed_data = {"premises": [], "hypotheses": [], "labels": []}

    for i, premise in enumerate(data['premises']):
        # Ignore sentences that have a label for which no index was
        # defined in 'labeldict'.
        label = data["labels"][i]
        if label not in labeldict:
            print('error label')
            continue

        transformed_data["labels"].append(labeldict[label])

        indices = words_to_indices(premise, worddict)
        transformed_data["premises"].append(indices)

        indices = words_to_indices(data["hypotheses"][i], worddict)
        transformed_data["hypotheses"].append(indices)

    return transformed_data


def build_embedding_matrix(worddict, embeddings_file):
    """
    Build an embedding matrix with pretrained weights for a given worddict.

    Args:
        worddict: A dictionary associating words to unique integer indices.
        embeddings_file: A file containing pretrained word embeddings.

    Returns:
        A numpy matrix of size (num_words+4, embedding_dim) containing
        pretrained word embeddings (the +4 is for the padding, BOS, EOS and
        out-of-vocabulary tokens).
    """
    # Load the word embeddings in a dictionnary.
    embeddings = {}
    seen_word = 0
    with open(embeddings_file, 'r', errors="ignore") as input_data:
        for line in input_data:
            line = line.rstrip().split(' ')

            try:
                # Check that the second element on the line is the start
                # of the embedding and not another word. Necessary to
                # ignore multiple word lines.
                float(line[1])
                word = line[0]
                if word in worddict:
                    embeddings[word] = line[1:]

            # Ignore lines corresponding to multiple words separated
            # by spaces.
            except ValueError:
                print('error')
                continue

    num_words = len(worddict)
    embedding_dim = len(list(embeddings.values())[0])
    embedding_matrix = np.zeros((num_words, embedding_dim))
    print('num_words: %s, emb_dim: %s' % (num_words, embedding_dim))
    # Actual building of the embedding matrix.
    for word, i in worddict.items():
        if word in embeddings:
            seen_word += 1
            embedding_matrix[i] = np.array(embeddings[word], dtype=float)
        else:
            if word == "_PAD_":
                continue
            # Out of vocabulary words are initialised with random gaussian
            # samples.
            embedding_matrix[i] = np.random.normal(size=(embedding_dim))
    print('share words: %s' % seen_word)
    return embedding_matrix


def preprocess_NLI_data(inputdir,
                        embeddings_file,
                        targetdir,
                        lowercase=False,
                        ignore_punctuation=False,
                        num_words=None,
                        datatype='NLI'):
    """
    Preprocess the data from some NLI corpus so it can be used by the
    ESIM model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        print(file)
        if fnmatch.fnmatch(file, '*-train.*'):
            train_file = file
        elif fnmatch.fnmatch(file, '*-dev.*'):
            dev_file = file
        elif fnmatch.fnmatch(file, '*-test.*'):
            test_file = file

    # -------------------- Train data preprocessing -------------------- #
    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    if datatype == "NLI":
        data = read_data(os.path.join(inputdir, train_file),
                         lowercase=lowercase,
                         ignore_punctuation=ignore_punctuation)
    elif datatype == 'Sogou':
        data = read_data_sogou(os.path.join(inputdir, train_file),
                               lowercase=False,
                               ignore_punctuation=False
                               )
    else:
        raise NotImplementedError
    print("\t* Computing worddict and saving it...")
    worddict = build_worddict(data, num_words=num_words)
    with open(os.path.join(targetdir, "worddict.pkl"), 'wb') as pkl_file:
        pickle.dump(worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    if datatype == 'NLI':
        labeldict = {"entailment": 0, "neutral": 1, "contradiction": 2}
    elif datatype == 'Sogou':
        labeldict = {0: 0, 1: 1}
    else:
        raise NotImplementedError

    transformed_data = transform_to_indices(data, worddict, labeldict)
    print(len(transformed_data))
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    if datatype == "NLI":
        data = read_data(os.path.join(inputdir, dev_file),
                         lowercase=lowercase,
                         ignore_punctuation=ignore_punctuation)
    elif datatype == 'Sogou':
        data = read_data_sogou(os.path.join(inputdir, dev_file),
                               lowercase=False,
                               ignore_punctuation=False
                               )
    else:
        raise NotImplementedError

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform_to_indices(data, worddict, labeldict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("\t* Reading data...")

    if datatype == "NLI":
        data = read_data(os.path.join(inputdir, test_file),
                         lowercase=lowercase,
                         ignore_punctuation=ignore_punctuation)
    elif datatype == 'Sogou':
        data = read_data_sogou(os.path.join(inputdir, test_file),
                               lowercase=False,
                               ignore_punctuation=False
                               )
    else:
        raise NotImplementedError

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_data = transform_to_indices(data, worddict, labeldict)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), 'wb') as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = build_embedding_matrix(worddict, embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), 'wb') as pkl_file:
        pickle.dump(embed_matrix, pkl_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess an NLI dataset')
    parser.add_argument('--config',
                        default="../config/preprocessing.json",
                        help='Path to a configuration file for preprocessing')
    args = parser.parse_args()

    with open(os.path.normpath(args.config), 'r') as cfg_file:
        config = json.load(cfg_file)

    preprocess_NLI_data(os.path.normpath(config["data_dir"]),
                        os.path.normpath(config["embeddings_file"]),
                        os.path.normpath(config["target_dir"]),
                        lowercase=config["lowercase"],
                        ignore_punctuation=config["ignore_punctuation"],
                        num_words=config["num_words"],
                        datatype=config.get('datatype', 'NLI')
                        )
