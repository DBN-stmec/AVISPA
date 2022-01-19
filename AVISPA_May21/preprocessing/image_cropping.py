import cv2
import os

#THIS CODE takes a folder (FILE_PATH) of images to crop it to the size: 299x299. The cropped images are written into a new folder (OUTPUT_PATH)

# Input folder, where the images should be cropped
FILE_PATH = r"C:\Users\DB\Studium\SoSe21\Hiwi_ML_PTW\AVISPA\TestCase\data\RNGN19cross\validation\OK"
# Output folder, where the cropped images are stored
OUTPUT_PATH = r"C:\Users\DB\Studium\SoSe21\Hiwi_ML_PTW\AVISPA\TestCase\data\RNGN19cross\validation\OK_cropped"


def crop(image_path:str, dict):
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    center_x, center_y = round(width / 2), round(height / 2)

    x = center_x - dict['delta_x']
    y = center_y - dict['delta_y']
    h = dict['delta_h']
    w = dict['delta_w']

    cropped_img = img[y:y + h, x:x + w]
    resized_img = resizeImage(cropped_img)
    return resized_img

def resizeImage(img):
    return cv2.resize(img, (299, 299))

def write_image(image, original_file):
    # split file path to get file name
    splits = str(original_file).split("\\")
    filename = splits[-1]
    new_filepath = OUTPUT_PATH + r"\\" + filename
    cv2.imwrite(new_filepath,image)

def get_files(target_dir):
    item_list = os.listdir(target_dir)

    file_list = list()
    for item in item_list:
        item_dir = os.path.join(target_dir,item)
        if os.path.isdir(item_dir):
            file_list += get_files(item_dir)
        else:
            file_list.append(item_dir)
    return file_list

# Get the camera angle
def getSub(target_dir:str) -> int :
    subNum = 0
    if 'sub2' in target_dir:
        subNum = 2
    elif 'sub4' in target_dir:
        subNum = 4
    elif 'sub5' in target_dir:
        subNum = 5
    elif 'sub6' in target_dir:
        subNum = 6
    elif 'sub8' in target_dir:
        subNum = 8
    return subNum

def getComponentID(target_path:str) -> str:
    id = ""
    if 'RNGN19' in target_path:
        id = 'RNGN19'
    elif 'RCGX12' in target_path:
        id = 'RCGX12'
    elif 'RNGN12' in target_path:
        id = 'RNGN12'
    return id

CONFIG={
    'RNGN19':{
        0: {"delta_x": 1, "delta_y": 0, "delta_h": 0, "delta_w": 0},
        2: {"delta_x": 800, "delta_y": 700, "delta_h": 1500, "delta_w": 1500},
        4: {"delta_x": 1000, "delta_y": 550, "delta_h": 1500, "delta_w": 1500},
        5: {"delta_x": 700, "delta_y": 500, "delta_h": 1500, "delta_w": 1500},
        6: {"delta_x": 400, "delta_y": 500, "delta_h": 1500, "delta_w": 1500},
        8: {"delta_x": 900, "delta_y": 400, "delta_h": 1800, "delta_w": 1900}
    },
    'RCGX12':{
        0: {"delta_x": 0, "delta_y": 0, "delta_h": 0, "delta_w": 0}},
    'RNGN12':{
        0:{"delta_x": 0, "delta_y": 0, "delta_h": 0, "delta_w": 0}},
}


if __name__=="__main__":
    file_list = get_files(FILE_PATH)
    for file_path in file_list:
        id = getComponentID(file_path)
        subNum = getSub(file_path)
        cropped = crop(file_path, CONFIG[id][subNum])
        write_image(cropped, file_path)
