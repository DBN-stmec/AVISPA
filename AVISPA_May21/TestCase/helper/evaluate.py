import numpy
import stats
import config
from sklearn.metrics import precision_recall_curve, roc_curve, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from classify import util
import pickle

labels = {}


def print_summary(args):
    print("")
    print("---------- SUMMARY ----------")
    print("Total images: %d" % stats.get('total_count', 0))

    if not args.skip_detection:
        print("Detected objects: %d (%d%%)" % (
            stats.get('detected_count', 0),
            round(100 * stats.get('detected_count', 0)/stats.get('total_count', 1))))

    print("Classified objects: %d (%d%%)" % (
        stats.get('classified_count', 0),
        round(100 * stats.get('classified_count', 0) / stats.get('total_count', 1))))

    print("Labels: %s" % config.CLASSIFY['classes'])

    y_true = stats.get('y_true')
    y_pred = stats.get('y_pred')

    if args.expected and stats.get('classified_count') is not None:
        dump_metrics()

        print("Confusion matrix:")
        print(get_confusion_matrix(y_true, y_pred))

        print(get_classification_report(y_true, y_pred))
    else:
        print("Result: %s" % get_label_counts(y_pred))

    print("Average processing time: %sms" % round(stats.get_average('processing_time', 0)))

    if not args.skip_detection:
        print("Average detection time: %sms" % round(stats.get_average('detection_time', 0)))

    print("Average classification time: %sms" % round(stats.get_average('classification_time', 0)))
    print("Average crop time: %sms" % round(stats.get_average('crop_time', 0) * 1000))

    fps = stats.get('frame_count', 0) / (stats.get('end_time') - stats.get('start_time'))
    print("Frames per second: %.1f" % fps)
    print("Time passed: %.2fs" % (stats.get('end_time') - stats.get('start_time')))
    print("Time per frame: %.4fs" % ((stats.get('end_time') - stats.get('start_time')) / stats.get('frame_count')))
    print("-----------------------------")


def get_result():
    label_scores = {}
    for label in config.CLASSIFY['classes']:
        label_scores[label] = {
            "label": label,
            "count": 0,
            "score": 0
        }

    if config.GENERAL['mode'] == 'floating':
        results = stats.get_last('classification_results', config.CLASSIFY['status_result_count'])
    else:
        results = stats.get('classification_results', [])
    for result in results:
        label = result["label"]
        score = float(result["score"])
        label_scores[label]["count"] += 1
        label_scores[label]["score"] += score

    score = 0
    prediction = None
    count = 1
    for label in label_scores:
        if label_scores[label]["score"] > score:
            prediction = label_scores[label]["label"]
            score = label_scores[label]["score"]
            count = label_scores[label]["count"]

    score /= count

    OK_count = label_scores["OK"]["count"]
    NOK_count = label_scores["NOK"]["count"]

    return prediction, score, OK_count, NOK_count


def print_status():
    prediction, score, OK_count, NOK_count = get_result()

    print("Frames: %d | Detected: %d | Classified: %d      \n"
          "Prediction: %s | Ok: %d | Verschlissen: %d      \n"
          "Detection time: %dms | Classification time: %dms      \n"
          "Wrong aspect ratio: %d | Too small: %d | Too bright: %d | Too dark: %d | Too blurry: %d (%d)\n" %
          (stats.get('frame_count', 0),
           stats.get('detected_count', 0),
           stats.get('classified_count', 0),
           prediction,
           OK_count,
           NOK_count,
           stats.get_average('detection_time', 0),
           stats.get_average('classification_time', 0),
           stats.get("aspect_ratio_wrong", 0),
           stats.get("small", 0),
           stats.get("bright", 0),
           stats.get("dark", 0),
           stats.get("blurry", 0),
           get_sharpness_threshold()
           ), end="\r\r\r\r")


def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_precision(y_true, y_pred):
    return precision_score(y_true, y_pred)


def get_recall(y_true, y_pred):
    return recall_score(y_true, y_pred)


def get_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


def get_label_counts(y_pred):
    return [y_pred.count(0), y_pred.count(1)]


def get_classification_report(y_true, y_pred):
    target_names = ['OK', 'NOK']

    print("Accuracy: %.2f" % get_accuracy(y_true, y_pred))
    print("Precision: %.2f" % get_precision(y_true, y_pred))
    print("Recall: %.2f" % get_recall(y_true, y_pred))
    print("F1: %.2f" % get_f1(y_true, y_pred))
    print("Metrics per class:")

    return classification_report(y_true, y_pred, target_names=target_names)


def get_sharpness_threshold():
    sharpness_values = stats.get('sharpness', [])
    if config.CLASSIFY['evaluate_sharpness'] and len(sharpness_values):
        threshold = numpy.percentile(sharpness_values, config.DETECT['sharpness_percentile'])
    else:
        threshold = 0
    return threshold


def log_classification(label, score, correct_label=None):
    stats.append('classification_results', {
        "label": label,
        "score": score
    })

    global labels
    if label not in labels:
        labels[label] = {
            "correct": 0,
            "wrong": 0,
            "count": 0
        }

    if correct_label is not None:
        if label == correct_label:
            labels[label]["correct"] += 1
        else:
            labels[label]["wrong"] += 1
    labels[label]["count"] += 1

    stats.increment("classified_count")

    y_true = False
    for i in range(len(config.CLASSIFY['classes'])):
        if config.CLASSIFY['classes'][i] == correct_label:
            y_true = i

    stats.append('y_true', y_true)


def colorize(string, color=None):
    string = str(string)
    if color == 'green':
        color_string = '\033[92m'
    elif color == 'red':
        color_string = '\033[91m'
    else:
        color_string = '\033[94m'

    end_color_string = '\033[0m'
    return color_string + string + end_color_string


def reject_outliers(data, m = 3.):
    d = numpy.abs(data - numpy.median(data))
    mdev = numpy.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def dump_metrics():
    y_true = stats.get('y_true')
    y_score = stats.get('y_score')
    y_pred = stats.get('y_pred')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_score)

    metrics_filename = util.get_named_model_path() + '/metrics.dict'

    with open(metrics_filename, 'wb') as metrics_file:
        pickle.dump((precision, recall, true_positive_rate, false_positive_rate, thresholds, y_true, y_score, y_pred), metrics_file)
        print("Dumped metrics to %s" % metrics_filename)