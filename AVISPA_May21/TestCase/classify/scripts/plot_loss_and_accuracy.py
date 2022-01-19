import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle
import argparse
import config
from classify import util
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--model_names", required=True, nargs='+')
parser.add_argument("--labels", nargs='+')
parser.add_argument("--limit", default=999999, type=int)
args = parser.parse_args()

fig, axes = plt.subplots(nrows=len(args.model_names), ncols=2, figsize=(8.27, 10.69), dpi=100)

if not args.labels:
    labels = args.model_names
else:
    labels = args.labels

color_blue = (185 / 255, 15 / 255, 34 / 255, 1)
color_red = (79 / 255, 129 / 255, 189 / 255, 1)

i = 0
for model_name in args.model_names:
    config.CLASSIFY['model_name'] = model_name
    warmup_history_file_path = util.get_history_path("warmup")
    history_file_path = util.get_history_path()
    warmup_history_file = open(warmup_history_file_path, "rb")
    history_file = open(history_file_path, "rb")
    warmup_history = pickle.load(warmup_history_file)
    history = pickle.load(history_file)

    limit = args.limit
    min_val_loss = min(history['val_loss'])
    print("Model: %s" % model_name)
    print("Min validation loss: %.8f after %d epochs" % (min_val_loss, history['val_loss'].index(min_val_loss)))
    max_val_accuracy = max(history['val_accuracy'])
    print("Max validation accuracy: %.8f after %d epochs" % (max_val_accuracy, history['val_accuracy'].index(max_val_accuracy)))

    if len(args.model_names) > 1:
        axis1 = axes[i, 0]
        axis2 = axes[i, 1]
    else:
        axis1 = axes[0]
        axis2 = axes[1]

    axis1.set_title("%s" % labels[i])
    axis1.set_xlabel("Epoche")
    axis1.set_ylabel("Loss")
    axis1.plot(warmup_history['loss'][:limit] + history['loss'][:limit], color=color_blue)
    axis1.plot(warmup_history['val_loss'][:limit] + history['val_loss'][:limit], color=color_red)
    axis1.legend(['Training', 'Validierung'], loc='upper right')

    axis2.set_title("%s" % labels[i])
    axis2.set_xlabel("Epoche")
    axis2.set_ylabel("Accuracy")
    axis2.plot(warmup_history['accuracy'][:limit] + history['accuracy'][:limit], color=color_blue)
    axis2.plot(warmup_history['val_accuracy'][:limit] + history['val_accuracy'][:limit], color=color_red)
    axis2.legend(['Training', 'Validierung'], loc='lower right')

    i+=1

plt.tight_layout(pad=0.7)
plt.savefig("plots/loss_acc_%s.png" % ("_".join(args.model_names)))
plt.show()