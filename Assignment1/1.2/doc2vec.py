import json
import os
import pickle
import sys
import gensim
from tqdm import tqdm
import multiprocessing as mp

model_name = "models/doc2vec"


class LabeledLineSentence(object):
    def __init__(self, doc_list):
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield gensim.models.doc2vec.LabeledSentence(doc.split(), [idx])

if __name__ == '__main__':
    if os.path.exists(model_name):
        print("Loading Doc2vec model")
        model = gensim.models.Doc2Vec.load(model_name)
    else:
        print("\nReading Data\n")
        data = []
        with open("../dataset/audio_train.json", "r") as f:
            for line in tqdm(f):
                data.append(json.loads(line))

        print("\nExtracting x's and y's\n")
        corpus = [(each["summary"] + " ") * 4 + each["reviewText"] for each in data]
        y = [each["overall"] for each in data]
        it = LabeledLineSentence(corpus)

        model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=8, alpha=0.025, min_alpha=0.025)

        model.build_vocab(it)

    #training of model
    print("Learning doc2vec")
    for epoch in range(10):
        print("iteration "+str(epoch+1))
        model.train(it, total_examples=model.corpus_count, epochs=1)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    #saving the created model
    model.save(model_name)
    print("model saved")
