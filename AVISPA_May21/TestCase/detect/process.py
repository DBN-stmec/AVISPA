import sys
sys.path.append('../')
import random
import os
import datetime
import cv2
from detect.detector import Detector
from PIL import Image, ImageOps
import argparse
import xml.etree.ElementTree as ET
import config
import data_collector
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True,
               help='Can be a directory with images, a path to a single image or video file or "webcam"')
parser.add_argument("--input_scale", default=1, type=float)
parser.add_argument("--output_scale", default=1, type=float)
args = parser.parse_args()

detector = None


def init_engine():
    global detector
    detector = Detector()


def get_images_from_directory(path):
    files = os.listdir(path)
    if len(files) == 0:
        print("No test images found")
        exit()

    return files


def process_random_image(path):
    file = os.path.join(args.pathIn, random.choice(path))
    return process_image(file)


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3


# box = [ymin, xmin, ymax, xmax]
def get_bbox_iou(box1, box2):
    y1_min, x1_min, y1_max, x1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    box1_w = x1_max - x1_min
    box1_h = y1_max - y1_min
    box2_w = x2_max - x2_min
    box2_h = y2_max - y2_min

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1_w * box1_h + box2_w * box2_h - intersect

    return float(intersect) / union


def get_ground_truth_box(file):
    filename, extension = os.path.splitext(file)
    annotations_file = file.replace(extension, '.xml')
    annotations_file = annotations_file.replace('images', 'annotations')
    tree = ET.parse(annotations_file)
    xml = tree.getroot()
    width = float(xml.findtext('./size/width'))
    height = float(xml.findtext('./size/height'))
    xmin = float(xml.findtext('./object/bndbox/xmin'))
    ymin = float(xml.findtext('./object/bndbox/ymin'))
    xmax = float(xml.findtext('./object/bndbox/xmax'))
    ymax = float(xml.findtext('./object/bndbox/ymax'))

    #return xmin, ymin, xmax-xmin, ymax-ymin
    return ymin/height, xmin/width, ymax/height, xmax/width


def process_file(path):
    filename, extension = os.path.splitext(path)
    if extension == '.jpg':
        return process_image(path)
    elif extension == '.mp4':
        return process_video(path)


def process_image(path):
    image_processed = detector.process_image(path)
    image_processed.show()


def detect(image):
    scores, boxes, classes = detector.process_image(image)
    score = scores[0]
    box = boxes[0]

    return score, box


def process_all_images(path):
    files = get_images_from_directory(path)
    results = []

    for file in files:
        output = file
        file = os.path.join(path, file)

        if config.DETECT['benchmark']:
            start_time = datetime.datetime.now()

        # Do the actual detection
        image = cv2.imread(file)
        #image = detector.preprocess(image)
        first_score, first_box = detect(image)

        if config.DETECT['benchmark']:
            processing_time = datetime.datetime.now() - start_time
            data_collector.add_value('processing_time', int(processing_time.total_seconds() * 1000))

        has_matches = first_score > .5
        if not has_matches:
            output += ' - no match'
            if config.DETECT['benchmark']:
                data_collector.increment('false_negatives')
            continue
        else:
            if config.DETECT['benchmark']:
                ground_truth_box = get_ground_truth_box(file)
                iou = get_bbox_iou(ground_truth_box, tuple(first_box))

                # Collect data
                data_collector.add_value('iou', iou)
                data_collector.increment('detection_count')
                if iou > float(config.DETECT['iou_threshold']):
                    data_collector.increment('true_positives')
                else:
                    data_collector.increment('false_positives')

            score_percentage = 100 * first_score
            output += ' - %d%%' % score_percentage

            result = {
                "score": first_score,
                "file": file,
                "box": first_box
            }
            results.append(result)

        print(output)

    if config.DETECT['benchmark']:
        print('AP: {0:.2f}'.format(get_average_precision(
            data_collector.get('true_positives'),
            data_collector.get('false_positives', 0)
        )))
        print('Average IOU: {0:.3f}'.format(data_collector.get_average('iou')))
        print('Average detection time: {0:.1f} ms'.format(data_collector.get_average('detection_time')))
        print('Average processing time: {0:.1f} ms'.format(data_collector.get_average('processing_time')))
    return results


def get_average_precision(truePositives, falsePositives):
    if truePositives == 0 & falsePositives == 0:
        return 0.0

    return truePositives / (truePositives + falsePositives)


def process_and_crop_all_images():
    min_cropped_height = 90
    min_cropped_width = 90
    aspect_ratio_delta = 0.2

    results = process_all_images()
    for result in results:
        file = result["file"]
        image = Image.open(file)
        ymin, xmin, ymax, xmax = result["box"]
        width, height = image.size

        ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)

        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        width, height = cropped_image.size
        aspect_ratio = width / height

        if width > min_cropped_width and height > min_cropped_height and aspect_ratio > (1-aspect_ratio_delta) and aspect_ratio < (1+aspect_ratio_delta):
            #cropped_image.show()
            filename = os.path.basename(file)
            cropped_image = ImageOps.autocontrast(cropped_image)
            cropped_image.save(os.path.join(config.DETECT['export_dir'], filename))


def process_video(path):
    vidcap = cv2.VideoCapture(path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success = True

    if path != 0:
        print("Extracting frames from %s (%d frames total)" % (path, length))

    with Detector() as detector:
        while success:
            success, image_np = vidcap.read()

            if success:

                if config.DETECT['benchmark']:
                    start_time = datetime.datetime.now()

                scores, boxes, classes = detector.process_image(image_np)

                if config.DETECT['benchmark']:
                    processing_time = datetime.datetime.now() - start_time
                    data_collector.add_value('processing_time', int(processing_time.total_seconds() * 1000))

                processed_image = draw_boxes(image_np, scores, boxes, classes)

                # Show the processed image
                cv2.imshow('detection',
                           cv2.resize(processed_image, (0, 0),
                           fx=args.output_scale,
                           fy=args.output_scale))

                # Keep rendering images until 'q' is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

    if config.DETECT['benchmark']:
        print('Average detection time: {0:.1f} ms'.format(data_collector.get_average('detection_time')))
        print('Average processing time: {0:.1f} ms'.format(data_collector.get_average('processing_time')))

def process_webcam():
    # cv2 uses the first available camera if path=0
    print("Processing webcam")
    process_video(0)


def draw_boxes(image_np, scores, boxes, classes, max_boxes=1, thickness=4, color=(0, 225, 0), font_size=24):
    image = Image.fromarray(np.uint8(image_np)).convert('RGB')
    im_width, im_height = image.size
    for i in range(len(boxes)):

        if i >= max_boxes:
            break

        ymin, xmin, ymax, xmax = boxes[i]

        draw = ImageDraw.Draw(image)
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', font_size)
        except IOError:
            font = ImageFont.load_default()

        # Since we only have one class - Schneide - only show this Label
        # Darknet gives class = 0, tensorflow class = 1
        label = config.DETECT['label'] if classes[i] <= 1 else '???'
        display_str = '{}: {}%'.format(
            label,
            int(100*scores[i]))

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = font.getsize(display_str)
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height

        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


    np.copyto(image_np, np.array(image))
    #image_np = image

    return image_np


def main():
    init_engine()
    if args.source == "webcam":
        process_webcam()
    elif os.path.isdir(args.source):
        process_all_images(args.source)
    elif os.path.isfile(args.source):
        process_file(args.source)
    else:
        print('Ungültige Quelle')


if __name__ == "__main__":
    main()