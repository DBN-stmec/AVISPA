import cv2
import config

options = {}

def initialize_windows(args):
    global options
    options = args

    if options.show_detected:
        cv2.namedWindow('detected')

    if options.show_classified:
        cv2.namedWindow('classified')

    if options.show_cropped:
        cv2.namedWindow('cropped')

    if options.show_raw:
        cv2.namedWindow('raw')


def show(name, image, wait=1):
    global options
    if window_wanted(name) and image is not None:
        image = cv2.resize(image, (0, 0,), fx=config.VIDEO_OUTPUT['scale'], fy=config.VIDEO_OUTPUT['scale'])
        cv2.imshow(name, image)
        if options.wait:
            cv2.waitKey(0)
        elif wait is not None:
            cv2.waitKey(wait)


def window_wanted(name):
    global options
    if name == 'detected':
        return options.show_detected
    elif name == 'classified':
        return options.show_classified
    elif name == 'cropped':
        return options.show_cropped
    elif name == 'raw':
        return options.show_raw