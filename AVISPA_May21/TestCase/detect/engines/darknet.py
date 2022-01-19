from ctypes import *
import os
import datetime
from detect.engine import Engine
import data_collector
import cv2
import config

DARKNET_PATH = '/home/jonas/code/darknet'
BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
MODELS_PATH = os.path.join(BASE_PATH, 'models', 'darknet')

class DarknetEngine(Engine):

    class BOX(Structure):
        _fields_ = [("x", c_float),
                    ("y", c_float),
                    ("w", c_float),
                    ("h", c_float)]

    class IMAGE(Structure):
        _fields_ = [("w", c_int),
                    ("h", c_int),
                    ("c", c_int),
                    ("data", POINTER(c_float))]

    class METADATA(Structure):
        _fields_ = [("classes", c_int),
                    ("names", POINTER(c_char_p))]

    def __init__(self, model_name):
        self.model_name = model_name

        self.init_engine()

        Engine.__init__(self)
        return

    def init_engine(self):
        lib = CDLL(os.path.join(DARKNET_PATH, 'libdarknet.so'), RTLD_GLOBAL)
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int
        self.lib = lib

        predict = lib.network_predict
        predict.argtypes = [c_void_p, POINTER(c_float)]
        predict.restype = POINTER(c_float)
        self.predict = predict

        set_gpu = lib.cuda_set_device
        set_gpu.argtypes = [c_int]

        make_image = lib.make_image
        make_image.argtypes = [c_int, c_int, c_int]
        make_image.restype = self.IMAGE

        make_boxes = lib.make_boxes
        make_boxes.argtypes = [c_void_p]
        make_boxes.restype = POINTER(self.BOX)
        self.make_boxes = make_boxes

        free_ptrs = lib.free_ptrs
        free_ptrs.argtypes = [POINTER(c_void_p), c_int]
        self.free_ptrs = free_ptrs

        num_boxes = lib.num_boxes
        num_boxes.argtypes = [c_void_p]
        num_boxes.restype = c_int
        self.num_boxes = num_boxes

        make_probs = lib.make_probs
        make_probs.argtypes = [c_void_p]
        make_probs.restype = POINTER(POINTER(c_float))
        self.make_probs = make_probs

        detect = lib.network_predict
        detect.argtypes = [c_void_p, self.IMAGE, c_float, c_float, c_float, POINTER(self.BOX),
                           POINTER(POINTER(c_float))]

        reset_rnn = lib.reset_rnn
        reset_rnn.argtypes = [c_void_p]

        load_net = lib.load_network
        load_net.argtypes = [c_char_p, c_char_p, c_int]
        load_net.restype = c_void_p
        self.load_net = load_net

        free_image = lib.free_image
        free_image.argtypes = [self.IMAGE]
        self.free_image = free_image

        letterbox_image = lib.letterbox_image
        letterbox_image.argtypes = [self.IMAGE, c_int, c_int]
        letterbox_image.restype = self.IMAGE

        load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = self.METADATA
        self.load_meta = load_meta

        load_image = lib.load_image_color
        load_image.argtypes = [c_char_p, c_int, c_int]
        load_image.restype = self.IMAGE
        self.load_image = load_image

        rgbgr_image = lib.rgbgr_image
        rgbgr_image.argtypes = [self.IMAGE]

        predict_image = lib.network_predict_image
        predict_image.argtypes = [c_void_p, self.IMAGE]
        predict_image.restype = POINTER(c_float)
        self.predict_image = predict_image

        network_detect = lib.network_detect
        network_detect.argtypes = [c_void_p, self.IMAGE, c_float, c_float, c_float, POINTER(self.BOX),
                                   POINTER(POINTER(c_float))]
        self.network_detect = network_detect

        model_path = os.path.join(MODELS_PATH, self.model_name)
        cfg_path = os.path.join(model_path, self.model_name + ".cfg")
        weights_path = os.path.join(model_path, self.model_name + ".weights")
        data_path = os.path.join(model_path, self.model_name + ".data")

        self.net = self.load_net(bytes(cfg_path, "ascii"), bytes(weights_path, "ascii"), 0)
        self.meta = self.load_meta(bytes(data_path, "ascii"))

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(bytes(image, "ascii"), 0, 0)
        boxes = self.make_boxes(self.net)
        scores = self.make_probs(self.net)
        num = self.num_boxes(self.net)

        if config.DETECT['benchmark']:
            detection_start = datetime.datetime.now()
        self.network_detect(self.net, im, thresh, hier_thresh, nms, boxes, scores)
        if config.DETECT['benchmark']:
            detection_time = datetime.datetime.now() - detection_start
            data_collector.add_value('detection_time', int(detection_time.total_seconds() * 1000))

        res_classes = []
        res_scores = []
        res_boxes = []

        image = cv2.imread(image)
        image_height, image_width, channels = image.shape

        for j in range(num):
            for i in range(self.meta.classes):
                if scores[j][i] > 0:
                    res_classes.append(i)
                    res_scores.append(scores[j][i])
                    darknet_box = (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)
                    res_boxes.append(self.convert_box(darknet_box, image_width, image_height))


        res = res_scores, res_boxes, res_classes

        self.free_image(im)
        self.free_ptrs(cast(scores, POINTER(c_void_p)), num)
        return res

    def convert_box(self, box, image_width, image_height):
        x, y, width, height = box

        ymin = (y - height * 0.5) / image_height
        xmin = (x - width * 0.5) / image_width
        ymax = (y + height * 0.5) / image_height
        xmax = (x + width * 0.5) / image_width

        box = ymin, xmin, ymax, xmax

        return box

    def process_file(self, file):
        return self.detect(file)

    def process_image(self, image_np):
        tmp_filename = 'tmp.jpg'
        cv2.imwrite(tmp_filename, image_np)
        result = self.process_file(tmp_filename)
        os.remove(tmp_filename)
        return result
