import os
abspath = os.path.dirname(os.path.abspath(__file__))

GENERAL = {
    'base_path': abspath,
    'mode': 'floating' # absolute / floating
}

DETECT = {
    'detect_path': os.path.join(abspath, 'detect'),
    'engine': 'tensorflow',
    'model': 'ssd_mobilenet_v1_coco',
    'iou_threshold': 0.5,
    'benchmark': True,
    'tensorflow_object_detection_path': '/home/jonas/code/tensorflow-models/research/object_detection',
    'label': 'Schneide',
    'score_threshold': .95,
    'box_zoom': 1.10,
    'brightness_min': 70, # max = 255
    'brightness_max': 240, # max = 255
    'preprocess_translate_x': 800,
    'preprocess_translate_y': 200,
    'preprocess_zoom': 3.5,
    'sharpness_percentile': 65, # max = 100
    'crop_box_scale': .6, # only take the lower right 60% of the detected box
    'mask_top_left': False,
    'mask_left_scale': 0.7,
    'mask_top_scale': 0.7,
    'resize_to_classifier_input_size': False,
    'export_path': os.path.join(abspath, 'export'),
    'translate_xy_percentile': 85
}

CLASSIFY = {
    'classify_path': os.path.join(abspath, 'classify'),
    'engine': 'keras',
    'model': '',
    'model_name': '',
    'benchmark': False,
    'trained_dir': os.path.join(abspath, 'classify', 'trained'),
    'classes': ['OK', 'NOK'],
    'plots_dir': os.path.join(abspath, 'plots'),
	'log_dir' : os.path.join(abspath, 'log'),
    'activation_path': os.path.join(abspath, 'classify', 'trained', 'activations.csv'),
    'fine_tuned_weights_filename': 'fine-tuned-{}-weights.h5',
    'classes_filename': 'classes-{}',
    'models_filename': 'model-{}.h5',
    'history_filename': 'history.dict',
    'class_mode': 'binary', # or categorical
    'min_warmup_epochs': 5,
    'score_threshold': .5,
    'evaluate_sharpness': True,
    'sharpness_threshold_calculation_skip': 1,
    'sharpness_threshold_calculation_detection_count': 25,
    'translate_x_threshold_calculation_detection_count': 25,
    'status_result_count': 20,
    'train_augmentation': 0.15,
    'grayscale': False,
    'learning_rate': False
}

VIDEO_OUTPUT = {
    'width': 640,
    'height': 480,
    'scale': 1
}