import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
sys.path.append('../')
from classify import util
import config


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_img', dest='path_img', help='Path to image', required=True, default=None, type=str)
    parser.add_argument('--accuracy', action='store_false', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_false')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_false')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture')
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    parser.add_argument('--model_name', required=True)
    args = parser.parse_args()
    return args

#path_img = 'C:\\users\\a.ziegenbein_lokal\\desktop\\data\\test\\verschlissen'

def get_files(path_img):
    if os.path.isdir(path_img):
        files = glob.glob(os.path.join(path_img, '*.jpg'))
    elif path_img.find('*') > 0: 
        files = glob.glob(path_img)
    else: 
        files = [path_img]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        
        x = model_module.load_file(i)
        
        
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    return y_true, inputs


def predict(path_img):
    files = get_files(path_img)
    n_files = len(files)
    print('Found {} files'.format(n_files))

    y_trues = []
    predictions = np.zeros(shape=(n_files,))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)

        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true

        if args.store_activations:
            util.save_activations(model, inputs, files[n_from:n_to], model_module.noveltyDetectionLayerName, n)

        if not args.store_activations:
            # Warm up the model
            if n == 0:
                print('Warming up the model')
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                print('Warming up took {} s'.format(end - start))

            # Make predictions
            start = time.clock()
            out = model.predict(np.array(inputs))
            end = time.clock()
            predictions[n_from:n_to] = np.argmax(out, axis=1)
            print('Prediction on batch {} took: {}'.format(n, end - start))

    if not args.store_activations:
        for i, p in enumerate(predictions):
            recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
            print('| should be {} ({}) -> predicted as {} ({})'.format(y_trues[i], files[i].split(os.sep)[-2], p,
                                                                       recognized_class))

        if args.accuracy:
            print('Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions)))

        if args.plot_confusion_matrix:
            cnf_matrix = confusion_matrix(y_trues, predictions)
            util.plot_confusion_matrix(cnf_matrix, config.CLASSIFY['classes'], normalize=False)
            util.plot_confusion_matrix(cnf_matrix, config.CLASSIFY['classes'], normalize=True)


if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('=' * 50)

    config.CLASSIFY['data_dir'] = args.data_dir
    config.CLASSIFY['train_dir'] = os.path.join(args.data_dir, 'train/')
    config.CLASSIFY['validation_dir'] = os.path.join(args.data_dir, 'validation/')
    # TODO: This is weird, what comes first, cli param or config?
    config.CLASSIFY['model'] = args.model
    config.CLASSIFY['model_name'] = args.model_name

    util.set_img_format()
    model_module = util.get_model_class_instance()

    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    predict(args.path_img)

    if args.execution_time:
        toc = time.clock()
        print('Time: %s' % (toc - tic))