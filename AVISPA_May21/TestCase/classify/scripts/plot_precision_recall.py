import pickle
from classify import util
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import config
from helper import evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_names", required=True, nargs='+')
parser.add_argument("--labels", nargs='+')
args = parser.parse_args()

if not args.labels:
    labels = args.model_names
else:
    labels = args.labels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.27,4), dpi=100)
for i in range(len(args.model_names)):
    model = args.model_names[i]
    label = labels[i]
    config.CLASSIFY['model_name'] = model

    metrics_file_path = util.get_named_model_path() + "/metrics.dict"
    metrics_file = open(metrics_file_path, "rb")
    (precision, recall, true_positive_rate, false_positive_rate, thresholds, y_true, y_score, y_pred) = pickle.load(metrics_file)

    #ax1.fill_between(recall, precision, step='post', alpha=0.2)

    roc_auc = auc(false_positive_rate, true_positive_rate)
    ax1.step(recall, precision, where='post', label=label)
    print(label)
    print("Accuracy: %.2f" % evaluate.get_accuracy(y_true, y_pred))
    print("Precision: %.2f" % evaluate.get_precision(y_true, y_pred))
    print("Recall: %.2f" % evaluate.get_recall(y_true, y_pred))
    print("F1: %.2f" % evaluate.get_f1(y_true, y_pred))
    print("ROC AUC: %.2f" % roc_auc)
    ax2.plot(false_positive_rate, true_positive_rate, label=labels[i] + (" (AUC=%.3f)" % roc_auc))

ax1.set_title('Precision-Recall-Kurve')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.legend(loc="lower left")

ax2.plot([0, 1], [0, 1], lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_title('ROC')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')

ax2.legend(loc="lower right")
plt.tight_layout(pad=0.7)
plt.savefig("plots/precision_recall_roc_%s.png" % "_".join(args.model_names))
plt.show()