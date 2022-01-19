import os
import logging
logging.captureWarnings(True)
from detect.detector import Detector
from classify.classifier import Classifier
import argparse
import cv2
import numpy
import scipy.misc
from helper import image as image_helper, evaluate, window
import config
import stats
import glob
from classify import util
import time
import datetime
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
#GUI
from guizero import App, PushButton, Text, Slider, info
from guizero import App, Picture, Box

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help='Can be a directory with images, a path to a single image,'
                                                    ' video file or "webcam"')
parser.add_argument("--expected")
parser.add_argument("--export_mistakes", action="store_true")
parser.add_argument("--export_cropped", action="store_true")
parser.add_argument("--export_raw", action="store_true")
parser.add_argument("--show_detected", action="store_true")
parser.add_argument("--show_classified", action="store_true")
parser.add_argument("--show_cropped", action="store_true")
parser.add_argument("--show_raw", action="store_true")
parser.add_argument("--wait", action="store_true")
parser.add_argument("--cam", default=0, type=int)
parser.add_argument("--limit", default=0, type=int)
parser.add_argument("--skip", default=0, type=int)
parser.add_argument("--sharpness", default=0, type=int)
parser.add_argument("--log_level", default="info", choices=["info", "debug"])
parser.add_argument("--skip_detection", action="store_true")
parser.add_argument("--skip_classification", action="store_true")
parser.add_argument("--classification_model")
parser.add_argument("--classification_model_name")
parser.add_argument("--classification_score_threshold")
parser.add_argument("--sharpness_percentile")
args = parser.parse_args()


# Set log level
if args.log_level == "debug":
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
    # Reduce TensorFlow Log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if args.classification_model:
    config.CLASSIFY['model'] = args.classification_model
if args.classification_model_name:
    config.CLASSIFY['model_name'] = args.classification_model_name
if args.sharpness_percentile:
    config.DETECT['sharpness_percentile'] = args.sharpness_percentile
if args.classification_score_threshold:
    config.CLASSIFY['score_threshold'] = float(args.classification_score_threshold)

detector = None
classifier = None


# Initializes the classification and detection engines
def init_engines():
    global detector, classifier
    logging.info("Initializing engines")

    util.get_model_from_model_name()

    if not args.skip_detection:
        detector = Detector()
        logging.info("Initialized detection engine")
        print("detection_ACTIVATED")

    if not args.skip_classification:
        classifier = Classifier()
        logging.info("Initialized classification engine")
        print("classification_ACTIVATED")


# Exits the classification and detection engines
def exit_engines():
    global detector, classifier

    logging.info("Exiting engines")

    if not args.skip_detection:
        detector.exit()

    if not args.skip_classification:
        classifier.exit()


def init_stats():
    stats.set('frame_count', 0)
    stats.set('total_count', 0)


# Crops part of the image according to bounding box coordinates
def crop_image_from_box(image_np, box, min_dimension=(40, 40), ar_delta=(0.15, 0.15)):
    start_time = time.time()

    image = image_np

    y_min, x_min, y_max, x_max = box
    height, width, channels = image.shape

    stats.append('detection_x_min', int(x_min * width))
    stats.append('detection_y_min', int(y_min * height))
    stats.append('detection_width', int((x_max - x_min) * width))
    stats.append('detection_height', int((y_max - y_min) * height))

    # Apply zoom
    if config.DETECT['box_zoom'] != 1:
        zoom = config.DETECT['box_zoom']
        logging.debug("Applying zoom %s to box" % config.DETECT['box_zoom'])

        relative_width = (x_max-x_min)
        relative_height = (y_max-y_min)

        x_max += 0.5 * (zoom-1) * relative_width
        y_max += 0.5 * (zoom-1) * relative_height

        if x_max > 1:
            x_max = 1
        if y_max > 1:
            y_max = 1

    y_min, x_min, y_max, x_max = int(y_min * height), int(x_min * width), int(y_max * height), int(x_max * width)

    width = x_max - x_min
    height = y_max - y_min
    ar = width / height

    if width < min_dimension[0] or height < min_dimension[1]:
        stats.increment('small')
        logging.debug("Ignored, cropped image is too small | Dimensions: %d x %d" % (width, height))
        return None

    # Check aspect ratio
    if (1 - ar_delta[0]) < ar < (1 + ar_delta[1]):
        cropped_image = image[y_min:y_max, x_min:x_max]

        if config.DETECT["resize_to_classifier_input_size"]:
            cropped_image = scipy.misc.imresize(cropped_image, classifier.get_input_size())
            height, width = cropped_image.shape[:2]

        if config.CLASSIFY['grayscale']:
            cropped_image = image_helper.convert_to_grayscale(cropped_image)

        # Check image brightness and sharpness
        if check_image_condition(cropped_image):

            # Only use the bottom right part of the box, assuming this is where the damage is
            if config.DETECT["crop_box_scale"] != 1:
                y_min = int((config.DETECT["crop_box_scale"]) * height)
                x_min = int((1 - config.DETECT["crop_box_scale"]) * width)
                cropped_image = cropped_image[y_min:y_max, x_min:x_max]
                height, width = cropped_image.shape[:2]

            # Choose CLAHE or contrast stretching here
            #cropped_image = image_helper.adaptive_histogram_equalization(cropped_image)
            cropped_image = image_helper.contrast_stretching(cropped_image)

            stats.append('crop_time', time.time() - start_time)
            return cropped_image
    else:
        stats.increment('aspect_ratio_wrong')
        logging.debug("Ignored, aspect ratio is out of bounds | Aspect ratio: %s" % round(ar, 2))

    return None


# Runs detection
def detect(image):
    scores, boxes, classes = detector.process_image(image)
    score = scores[0]
    box = boxes[0]

    return score, box


# Runs classification
def classify(image):
    temp_filename = 'tmp.jpg'
    # Write to temporary file that can be opened by classifier
    cv2.imwrite(temp_filename, image)
    classify.recognized_class, classify.score = classifier.process_file(temp_filename)
    return classify.recognized_class, classify.score


# Check image for brightness and sharpness
def check_image_condition(image):
    brightness = image_helper.calculate_brightness(image)
    sharpness = image_helper.calculate_sharpness(image)
    stats.append('sharpness', sharpness)
    sharpness_threshold = evaluate.get_sharpness_threshold() if config.GENERAL['mode'] == 'floating' else args.sharpness

    if brightness < config.DETECT['brightness_min']:
        logging.debug("Ignoring, image is too dark | brightness: (%d)" % brightness)
        stats.increment("dark")
    elif brightness > config.DETECT['brightness_max']:
        logging.debug("Ignoring, image is too bright | brightness: (%d)" % brightness)
        stats.increment("bright")
    elif config.CLASSIFY['evaluate_sharpness'] and sharpness < sharpness_threshold:
        logging.debug("Ignoring, image is too blurry | Sharpness: (%d)" % sharpness)
        stats.increment("blurry")
    else:
        return True

    return False


# Returns true if there are images in this directory
def is_image_directory(path):
    os.chdir(path)
    return len(glob.glob("*.jpg")) > 0


# Processes all files in the given directory
def process_directory(path):
    files = os.listdir(path)
    if len(files) == 0:
        logging.error("No files found in directory")
        exit()

    first_file = os.path.join(path, files[0])

    if os.path.isdir(first_file):
        if is_image_directory(first_file):
            logging.info("Treating sub directories as labels")
            process_label_directory(path)
            evaluate.print_summary(args)
        else:
            for file in files:
                sub_path = os.path.join(path, file)
                process_directory(sub_path)
    else:
        process_image_directory(path)
        evaluate.print_summary(args)


# Processes a directory with subdirectories named after the corresponding labels
def process_label_directory(path):
    stats.reset()
    for label in os.listdir(path):
        label_path = os.path.join(path, label) + "/"

        args.expected = label
        process_image_directory(label_path)

# Process every file in a directory with images
def process_image_directory(path):
    files = os.listdir(path)
    logging.info("Processing images in %s" % path)
    if len(files) == 0:
        logging.error("No images found in directory")
        exit()

    args.sharpness = 1

    stats.set('frame_count', 0)
    stats.set('start_time', time.time())

    for filename in files:
        stats.increment('frame_count')
        stats.increment('total_count')

        file_path = os.path.join(path, filename)

        image = cv2.imread(file_path)

        logging.debug("Processing image: %s, dimensions %d x %d " % (filename, image.shape[0], image.shape[1]))

        export_filename = filename + "_%d" % stats.get('frame_count')

        process_image(
            image=image,
            export_filename=export_filename,
            correct_label=args.expected)

        # Stop processing if limit has been reached
        if args.limit and stats.get('frame_count') >= args.limit:
            break

    stats.set('end_time', time.time())


# Process a video file
def process_video(path):
    use_cam = isinstance(path, int)

    calculate_translate_x(path)

    if not args.sharpness and config.CLASSIFY['evaluate_sharpness']:
        calculate_initial_sharpness_threshold(path)

    success = True
    filename = os.path.basename(path) if not use_cam else "cam"

    video = cv2.VideoCapture(path)
    if use_cam:
        logging.info("Extracting frames from webcam")
    else:
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info("Extracting frames from %s (%d frames total)" % (path, length))

    stats.set('start_time', time.time())
    stats.set('frame_count', 0)
    stats.set('total_count', 0)
    stats.set('detected_count', 0)
    stats.set('classified_count', 0)

    while success:
        # Grab video frame
        success, image = video.read()

        if success:
            stats.increment('frame_count')
            stats.increment('total_count')
            if args.skip and stats.get('frame_count', 0) % args.skip != 0:
                logging.debug("Skipping")
                continue

            logging.debug("Processing frame %d, dimensions %d x %d " % (stats.get('frame_count'), image.shape[0], image.shape[1]))

            export_filename = filename + "_%d" % stats.get('frame_count')

            process_image(
                image=image,
                export_filename=export_filename,
                correct_label=args.expected)

            if use_cam:
                evaluate.print_status()
            else:
                print("%d%%" % (100 * stats.get('frame_count', 0) / length), end="\r")

        # Stop processing if limit has been reached
        if args.limit and stats.get('frame_count') >= args.limit:
            break

    stats.set('end_time', time.time())


# Process detection
def process_detection(image, export_filename="detected"):
    detection_score, detection_box = detect(image)
    cropped_image = None

    if detection_score >= config.DETECT['score_threshold']:
        logging.debug("Found object, score: %d%%" % int(detection_score * 100))
        stats.increment("detected_count")

        cropped_image = crop_image_from_box(image, detection_box)

        if cropped_image is not None:
            logging.debug("Cropped image dimensions: (%d x %d)" % (cropped_image.shape[0], cropped_image.shape[1]))

            if args.export_cropped:
                export_path = os.path.join(config.DETECT["export_path"], args.expected, export_filename + '.jpg')
                cv2.imwrite(export_path, cropped_image)

            if args.show_detected:
                draw_scores = [detection_score]
                draw_boxes = [detection_box]
                image_processed = detector.draw_boxes(image, draw_scores, draw_boxes)
                window.show('detected', image_processed, 1)
    else:
        window.show('detected', image, 1)

    return cropped_image, detection_box, detection_score


# Process a single frame
def process_image(image, export_filename="export", correct_label=None):
    processing_start = datetime.datetime.now()

    if args.skip_detection:
        cropped_image = image
    else:
        image = detector.preprocess(image)
        cropped_image, detection_box, detection_score = process_detection(image, export_filename)

    window.show('raw', image, 1)

    if args.export_raw:
        export_path = os.path.join(config.DETECT["export_path"], "raw", export_filename + '.jpg')
        cv2.imwrite(export_path, image)

    if cropped_image is not None:

        window.show('cropped', cropped_image, 1)

        if not args.skip_classification:

            label, classification_score = classify(cropped_image)
            evaluate.log_classification(label, classification_score, correct_label)

            logging.debug("Classified. label: %s | score: %d%%" % (label, int(classification_score * 100)))

            if args.expected is not None:
                if label != args.expected:
                    # Export images that have been not been classified correctly
                    if args.export_mistakes:
                        export_path = os.path.join(config.DETECT['export_path'], 'mistakes', export_filename + '_%d_%d.jpg' % (int(detection_score * 100), int(classification_score * 100)))
                        logging.debug("Wrong label. Exporting to %s" % export_path)
                        cv2.imwrite(export_path, cropped_image)

            if not args.skip_detection and args.show_classified:
                draw_boxes = [detection_box]

                if config.GENERAL['mode'] == 'floating':
                    label, classification_score, _, _ = evaluate.get_result()
                    draw_scores = [classification_score]
                    draw_labels = [label]
                else:
                    draw_scores = [classification_score]
                    draw_labels = [label]
                image_processed = detector.draw_boxes(image, draw_scores, draw_boxes, draw_labels)
                window.show('classified', image_processed, 1)

        if not args.sharpness:
            args.sharpness = evaluate.get_sharpness_threshold()

    processing_time = datetime.datetime.now() - processing_start
    stats.append("processing_time", processing_time.total_seconds() * 1000)



# Process a webcam video stream
def process_webcam(index=0):
    # cv2 uses the first available camera if path=0
    print("Processing webcam")
    process_video(index)


# Returns a list of class labels
def get_class_labels(classes):
    class_labels = []
    for i in range(len(classes)):
        class_labels[i] = classes[i]

    return class_labels


# Calculate the x position for the image cutoff by calculating a certain percentile of the x-positions of detected objects
def calculate_translate_x(path):
    video = cv2.VideoCapture(path)
    count = 0
    limit = config.CLASSIFY["translate_x_threshold_calculation_detection_count"]
    if limit <= 0:
        return

    logging.info("Calculating x-offset of object using the first %d detections" % limit)
    width, height = (0,0)

    success = True
    while success:
        success, image_np = video.read()
        if success:
            stats.increment('frame_count')
            if stats.get('frame_count', 0) % config.CLASSIFY["sharpness_threshold_calculation_skip"] != 0:
                continue

            if width == 0:
                width, height, channels = image_np.shape

            image_np = detector.preprocess(image_np)

            window.show('raw', image_np)

            detection_score, detection_box = detect(image_np)
            has_match = detection_score >= config.DETECT['score_threshold']

            if has_match:

                if args.show_detected:
                    draw_scores = [detection_score]
                    draw_boxes = [detection_box]

                    image_processed = detector.draw_boxes(image_np, draw_scores, draw_boxes)
                    window.show('detected', image_processed, 1)

                cropped_image = crop_image_from_box(image_np, detection_box)

                if cropped_image is not None:
                    count += 1

                    window.show('cropped', cropped_image)

                print("%d%%" % (100 * count / limit), end="\r")

                if limit and count >= limit:
                    break

    if not len(stats.get('detection_x_min', [])):
        exit('No objects detected')

    x_min_values = numpy.array(stats.get('detection_x_min'))
    translate_x = numpy.percentile(x_min_values, config.DETECT['translate_xy_percentile'])
    y_min_values = numpy.array(stats.get('detection_y_min'))
    translate_y = numpy.percentile(y_min_values, config.DETECT['translate_xy_percentile'])

    max_width = stats.get_average('detection_width')
    max_height = stats.get_average('detection_height')
    zoom = round(min(width / max_width, height/max_height), 2)
    logging.info("Calculated preprocessing: translate_x: %d translate_y: %d zoom: %s" % (translate_x, translate_y, zoom))
    config.DETECT['preprocess_translate_x'] = config.DETECT['preprocess_translate_x'] + int(translate_x)
    config.DETECT['preprocess_translate_y'] = config.DETECT['preprocess_translate_y'] + int(translate_y)
    config.DETECT['preprocess_zoom'] = float(zoom)


# Calculate the sharpness threshold. This will be recalculated after every frame.
def calculate_initial_sharpness_threshold(path):
    video = cv2.VideoCapture(path)
    count = 0
    limit = config.CLASSIFY["sharpness_threshold_calculation_detection_count"]
    stats.limit_count('sharpness', limit)

    logging.info("Calculating sharpness threshold using the first %d detections" % limit)

    success = True
    while success:
        success, image_np = video.read()
        if success:
            stats.increment('frame_count')
            stats.increment('total_count')
            if stats.get('frame_count', 0) % config.CLASSIFY["sharpness_threshold_calculation_skip"] != 0:
                continue

            image_np = detector.preprocess(image_np)

            window.show('raw', image_np)

            detection_score, detection_box = detect(image_np)
            has_match = detection_score >= config.DETECT['score_threshold']

            if has_match:

                if args.show_detected:
                    draw_scores = [detection_score]
                    draw_boxes = [detection_box]
                    image_processed = detector.draw_boxes(image_np, draw_scores, draw_boxes)
                    window.show('detected', image_processed, 1)

                cropped_image = crop_image_from_box(image_np, detection_box)

                if cropped_image is not None:
                    count += 1

                    window.show('cropped', cropped_image)

                print("%d%%" % (100 * count / limit), end="\r")

                if limit and count >= limit:
                    break
            else:
                window.show('detected', image_np)


# Create necessary directories
def create_directories():
    if args.expected:
        if args.export_cropped or args.export_mistakes:
            export_path = os.path.join(config.DETECT["export_path"], args.expected)
            if not os.path.isdir(export_path):
                os.mkdir(export_path)

    if args.export_raw:
        export_path = os.path.join(config.DETECT["export_path"], "raw")
        if not os.path.isdir(export_path):
            os.mkdir(export_path)


# Main routine
if __name__ == "__main__":
    args = parser.parse_args()


    init_engines()
    init_stats()
    window.initialize_windows(args)
    create_directories()



    if args.source == "webcam":
        process_webcam(args.cam)
        evaluate.print_summary(args)
    elif os.path.isfile(args.source):
        process_video(args.source)
        evaluate.print_summary(args)
    elif os.path.isdir(args.source):
        process_directory(args.source)
    else:
        logging.error("Invalid source")
        exit()

    exit_engines()


    #preperation for GUI
    label=classify.recognized_class
    colour = ""
    def boxcolour():
        if label == "OK":
            colour="green"
        if label == "NOK":
            colour = "red"
        return colour

    # GUI window
    app = App(title="Prediction_Results",height=600, width=500, layout="grid")

    model = Text(app, text="Modelname: " + args.classification_model, size=20, grid=[0, 1])
    dataset = Text(app, text="Image name: "+ os.listdir(args.source)[0], size=20, grid=[0, 2])
    scoretext = Text(app, text = "Training_score: %d%%" % int(classify.score * 100), size=20, grid=[0, 3])
    classified= Text(app, text="Prediction: %s" % label, color=boxcolour(), size=20, grid=[0, 4])
    blank = Text(app, text= "", grid=[0, 5], size=50)
    colourbox = Box(app, width="fill", visible=True, border=True, grid=[0, 6])
    colourbox.set_border(10, boxcolour())
    picture = Picture(colourbox, image = args.source+"/"+ os.listdir(args.source)[0])

    app.display()