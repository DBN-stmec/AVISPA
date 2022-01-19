import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import matplotlib.pyplot as plt
import csv


parser = argparse.ArgumentParser()
parser.add_argument("--model_names", required=True, nargs='+')
parser.add_argument("--labels", nargs='+')
parser.add_argument("--limit", default=999999, type=int)
args = parser.parse_args()

fig, axes = plt.subplots(nrows=len(args.model_names), ncols=1, figsize=(4.1, 4.1), dpi=100)

if not args.labels:
    labels = args.model_names
else:
    labels = args.labels

color_red = (185 / 255, 15 / 255, 34 / 255, 1)
color_blue = (79 / 255, 129 / 255, 189 / 255, 1)

i = 0
for model_name in args.model_names:
    history_file_path = "detect/models/tensorflow/" + model_name + "/loss.csv"
    print(history_file_path)

    steps = []
    loss = []

    with open(history_file_path, 'rt') as csvfile:
        rows = csv.reader(csvfile, delimiter=';', quotechar='"')
        next(rows, None) # Skip header
        for row in rows:
            time = row[0]
            steps.append(int(row[1]))
            loss.append(float(row[2]))

    limit = args.limit

    min_val_loss = min(loss)
    print("Model: %s" % model_name)
    print("Min validation loss: %.8f after %d steps" % (min_val_loss, steps[loss.index(min_val_loss)]))

    if len(args.model_names) > 1:
        axis1 = axes[i, 0]
    else:
        axis1 = axes

    axis1.set_title("%s" % labels[i])
    axis1.set_xlabel("Step")
    axis1.set_ylabel("Loss")
    axis1.plot(steps[:limit], loss[:limit], color=color_red)

    i+=1

plt.tight_layout(pad=0.7)
plt.savefig("plots/loss_%s.png" % ("_".join(args.model_names)))
plt.show()