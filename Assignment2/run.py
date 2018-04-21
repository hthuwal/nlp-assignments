import sys
import os
import sklearn_crfsuite
import numpy as np
import nltk
import pickle


def read_data(file):
    data = []
    with open(file, "r", errors='replace') as f:
        temp_data = []
        for line in f:
            if line.strip() == "":
                data.append(temp_data)
                temp_data = []
            else:
                temp_data.append(line.strip().split()[0])

    # return list of sentences
    return data


def w2f(sentence, i):

    word = sentence[i][0]
    postag = sentence[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'suffix4': word[-4:],
        'suffix3': word[-3:],
        'suffix2': word[-2:],
        'suffix1': word[-1:],
        'prefix4': word[0:4],
        'prefix3': word[0:3],
        'prefix2': word[0:2],
        'prefix1': word[0:1],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],

    }
    if i > 0:
        prev_word = sentence[i - 1][0]
        prev_tag = sentence[i - 1][1]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.isupper()': prev_word.isupper(),
            'prev_word_suffix[4]': prev_word[-4:],
            'prev_word_suffix[3]': prev_word[-3:],
            'prev_word_suffix[2]': prev_word[-2:],
            'prev_word_suffix[1]': prev_word[-1:],
            'prev_tag': prev_tag,
            'prev_tag[:2]': prev_tag[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        next_word = sentence[i + 1][0]
        next_tag = sentence[i + 1][1]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word.istitle()': next_word.istitle(),
            'next_word_prefix[4]': next_word[0:4],
            'next_word_prefix[3]': next_word[0:3],
            'next_word_prefix[2]': next_word[0:2],
            'next_word_prefix[1]': next_word[0:1],
            'next_word.isupper()': next_word.isupper(),
            'next_tag': next_tag,
            'next_tag[:2]': next_tag[:2],
        })
    else:
        features['EOS'] = True

    return features


def data2features(data):
    """ Convert Entire data to list of feature list """
    temp_data = []
    for sentence in data:
        temp_sentence = [w2f(sentence, i) for i in range(len(sentence))]
        temp_data.append(temp_sentence)

    return temp_data


def save2file(pred, outfile):
    with open(outfile, "w") as out:
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                out.write("%s %s\n" % (org_data[i][j], pred[i][j]))
            out.write("\n")


infile = sys.argv[1]
outfile = sys.argv[2]

print("Reading Test Data from %s" % (infile))
test_data = read_data(infile)
org_data = read_data(infile)

print("Performing POS tagging")
test_data = [nltk.pos_tag(sentence) for sentence in test_data]

print("Converting each sentence to list of features")
test_data = data2features(test_data)

if os.path.exists("crf.model"):
    print("Loading CRF Model")
    crf = pickle.load(open("crf.model", "rb"))
else:
    print("Model does not exist! Download model file first")
    sys.exit(0)

print("Predictions")
pred = crf.predict(test_data)

print("Saving to %s" % (outfile))
save2file(pred, outfile)

print("Done!!")
