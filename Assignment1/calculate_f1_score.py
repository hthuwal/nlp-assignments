import json
import sys
from sklearn.metrics import f1_score


gold_file = sys.argv[1]
pred_file = sys.argv[2]

print("Loading Test Set")
gold_label = []
with open(gold_file, "r") as f:
    i = 0
    for line in f:
        i += 1
        sys.stdout.write("\r\x1b[K" + "%d" % (i))
        sys.stdout.flush()

        temp = json.loads(line)
        gold_label.append(int(temp["overall"]))

print()
gold_label = [1 if i == 1 or i == 2 else 5 if i == 4 or i == 5 else 3 for i in gold_label]
pred_label = []

with open(pred_file, "r") as f:
    for line in f:
        pred_label.append(int(line.strip()))

print("\nMacro F1 score: %f" % (f1_score(gold_label, pred_label, average="macro")))
