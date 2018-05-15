import json
import pickle
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

print("Loading Test Set")
test_data = []
test_y = []
with open(input_file, "r") as f:
    i = 0
    for line in f:
        i += 1
        sys.stdout.write("\r\x1b[K" + "%d" % (i))
        sys.stdout.flush()

        temp = json.loads(line)
        test_data.append((temp["summary"] + " ") * 3 + temp["reviewText"])
        test_y.append(int(temp["overall"]))

print("Loading Model")
vectorizer, svc = pickle.load(open("2017MCS2074.model", "rb"))

print("Cacluating tfidf for dev data")
tfidf_test = vectorizer.transform(test_data)

del test_data

print("Predicting")
y_pred = svc.predict(tfidf_test)

y_pred = [1 if i == 1 or i == 2 else 5 if i == 4 or i == 5 else 3 for i in y_pred]

with open(output_file, "w") as f:
    for y in y_pred:
        f.write("%d\n" % y)
