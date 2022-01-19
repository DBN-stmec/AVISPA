from detect.engines.tensorflow import TensorflowEngine
from detect.engines.darknet import DarknetEngine
import numpy
import logging
import config
import datetime
import stats


class Detector:
    detection_graph = None
    sess = None
    model_name = None

    def __init__(self, model_name=config.DETECT["model"], engine=config.DETECT["engine"]):
        self.model_name = model_name
        self.model_type = engine
        self.engine = None
        self.init_engine()

    def init_engine(self):
        if self.model_type == "tensorflow":
            self.engine = TensorflowEngine(model_name=self.model_name)
        elif self.model_type == "darknet":
            self.engine = DarknetEngine(model_name=self.model_name)
        else:
            exit("Invalid engine")

    def process_file(self, file):
        scores, boxes, classes = self.engine.process_file(file)
        return scores, boxes, classes

    def process_image(self, image_np):
        detection_start = datetime.datetime.now()
        result = self.engine.process_image(image_np)
        detection_time = datetime.datetime.now() - detection_start
        stats.append("detection_time", detection_time.total_seconds() * 1000)
        return result

    def process_and_mark_image(self, image_np):
        return self.engine.process_and_mark_image(image_np)

    @staticmethod
    def preprocess(image_np):
        x_translate, y_translate = config.DETECT['preprocess_translate_x'], config.DETECT['preprocess_translate_y']
        zoom = config.DETECT['preprocess_zoom']

        if x_translate == 0 and y_translate == 0 and zoom == 1:
            return image_np

        height, width, channels = image_np.shape

        if x_translate >= width:
            logging.error("TRANSLATE_X is bigger than width")
            exit()

        if y_translate >= height:
            logging.error("TRANSLATE_Y is bigger than height")
            exit()

        height_zoomed = int(height / zoom)
        width_zoomed = int(width / zoom)

        width = max(height_zoomed, width_zoomed)

        y = y_translate
        x = x_translate
        cropped_image = image_np[y:y + width, x:x + width]

        return numpy.array(cropped_image)

    def draw_boxes(self, image_np, scores, boxes, classes=[]):
        return self.engine.draw_boxes(image_np, scores, boxes, classes)

    def exit(self):
        self.engine.exit()