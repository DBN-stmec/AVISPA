import numpy as np
import argparse
import traceback
import os
from classify import util
import config
#GUI
from guizero import App, PushButton, Text, Slider, info

np.random.seed(1337)  # for reproducibility

# Reduce TensorFlow Log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to data dir')
    parser.add_argument('--model', type=str, help='Base model architecture', choices=[
        'resnet50',
        'inception_v3',
        'vgg16',
        'vgg19',
        'xception',
        'inception_resnet_v2',
        'densenet121',
        'densenet169',
        'densenet201',
        'nasnet_large',
        'nasnet_mobile'])
    parser.add_argument('--nb_epoch', type=int, default=5)  #If no epoch is spezified, it will set: nb_epoch=5
    parser.add_argument('--freeze_layers_number', type=int, help='will freeze the first N layers and unfreeze the rest')
    parser.add_argument('--binary', action="store_true")
    parser.add_argument('--grayscale', action="store_true")
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--learning_rate')
    return parser.parse_args()


def init():
    util.set_img_format()
    util.override_keras_directory_iterator_next()
    util.set_classes_from_train_dir()
    util.set_samples_info()
    if not os.path.exists(config.CLASSIFY['trained_dir']):
        os.mkdir(config.CLASSIFY['trained_dir'])
    if not os.path.exists(util.get_named_model_path()):
        os.mkdir(util.get_named_model_path())


def train(nb_epoch, freeze_layers_number):
    model = util.get_model_class_instance(
        class_weight=util.get_class_weight(config.CLASSIFY['train_dir']),
        nb_epoch=nb_epoch,
        freeze_layers_number=freeze_layers_number)
    model.train()
    print('Training is finished!')


#GUI window (internal name: app)
app = App(title="Training-Parameters", width=1600, layout="grid")
args = parse_args()

modeltext = Text(app, text="Model:", size=20,  grid=[0,0])
datatext = Text(app, text="Dataset:", size=20,  grid=[0,1])
epochtext = Text(app, text="Epoch:", size=20,  grid=[0,2])
learningratetext = Text(app, text="Learning-Rate:", size=20,  grid=[0,3])

model = Text(app, text=args.model_name, size=30,  grid=[1,0])
data = Text(app, text=args.data_dir, size=30,  grid=[1,1])
epoch = Text(app, text=args.nb_epoch, size=30,  grid=[1,2])
learningrate = Text(app, text=str(args.learning_rate), size=30,  grid=[1,3])

if __name__ == '__main__':

    try:
        args = parse_args()
        config.CLASSIFY['data_dir'] = args.data_dir
        #Inside the dataset folder (example: TestCase/data/kaggle) are three other folders: train, validation and test. The desired folder ware picked with os.path.join(args.data_dir, 'train/')
        config.CLASSIFY['train_dir'] = os.path.join(args.data_dir, 'train/')  #The train path: 'train_dir' defined as /data/(...) + /train
        config.CLASSIFY['validation_dir'] = os.path.join(args.data_dir, 'validation/')
        config.CLASSIFY['model_name'] = args.model_name
        if args.learning_rate:
            config.CLASSIFY['learning_rate'] = float(args.learning_rate)
        if args.model:
            config.CLASSIFY['model'] = args.model
        # if args.binary:
        config.CLASSIFY['class_mode'] = "binary"
        if args.grayscale:
            config.CLASSIFY['class_mode'] = "grayscale"
        #Here the training process starts!
        init()
        train(args.nb_epoch, args.freeze_layers_number)
    except Exception as e:
        print(e)
        traceback.print_exc()

    app.display()
