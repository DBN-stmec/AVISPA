import tensorflow as tf
import os
import numpy as np
import datetime
from detect.engine import Engine
import data_collector
from PIL import Image, ImageFont, ImageDraw
import cv2
import config

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
MODELS_PATH = os.path.join(BASE_PATH, 'models', 'tensorflow')


class TensorflowEngine(Engine):

    def __init__(self, model_name, weights_file_name="frozen_inference_graph.pb"):
        self.model_name = model_name
        self.weights_file_name = weights_file_name

        self.detection_graph = None
        self.sess = None

        self.init_engine()
        Engine.__init__(self)

    def init_engine(self):
        detection_graph = tf.Graph()
        od_graph_def = tf.GraphDef()

        weights_file_path = os.path.join(MODELS_PATH, self.model_name, self.weights_file_name)

        with detection_graph.as_default():
            with tf.gfile.GFile(weights_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.detection_graph = detection_graph
            self.sess = tf.Session(graph=detection_graph)

    def detect(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        return boxes[0], scores[0], classes[0], num_detections

    def process_file(self, file):
        image_np = cv2.imread(file)
        # cv2 reads color channels as BGR, but we need RGB
        image_np = image_np[:, :, ::-1]

        # This works too, but is slow: cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        return self.process_image(image_np)

    def process_image(self, image_np):
        if config.DETECT['benchmark']:
            detection_start = datetime.datetime.now()
        (boxes, scores, classes, num_detections) = self.detect(image_np)
        if config.DETECT['benchmark']:
            detection_time = datetime.datetime.now() - detection_start
            data_collector.add_value('detection_time', int(detection_time.total_seconds() * 1000))
        return scores, boxes, classes

    def process_and_mark_image(self, image_np):
        return image_np

    def draw_boxes(self, image_np, scores, boxes, labels=[], max_boxes=1, thickness=4, color=(0, 225, 0), font_size=20):
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
                font_file_path = os.path.join(config.DETECT['detect_path'], '..', 'arial.ttf')
                font = ImageFont.truetype(font_file_path, font_size)
            except IOError:
                font = ImageFont.load_default()

            # Since we only have one class - Schneide - only show this Label
            # Darknet gives class = 0, tensorflow class = 1
            # label = config.get('Label') if classes[i] <= 1 else '???'
            if len(labels):
                label = labels[i]
                display_str = '{}: {}%'.format(
                    label,
                    int(100 * scores[i]))

                text_width, text_height = font.getsize(display_str)
                margin = np.ceil(0.05 * text_height)
                draw.rectangle(
                    [(0, 0), (left + text_width, text_height + 2 * margin)],
                    fill=color)
                draw.text(
                    (margin, margin),
                    display_str,
                    fill='black',
                    font=font)

                """
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
                """
        result = np.copy(image_np)
        np.copyto(result, np.array(image))
        # image_np = image

        return result

    def exit(self):
        try:
            self.sess.close()
            del self.sess
        except:
            pass
