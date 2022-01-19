import cv2
import os

#How to use:
# 1. Enter your input folder-directory with the images in "FILE_PATH"
# 2. Enter your output folder-directory for the cropped images(299x299) in "OUTPUT_PATH"
# !!!! Make sure that your directory has this seperator: "/", because // or :// won't work. In that cas you must change the seperator in the fnc: "write_image()"

FILE_PATH = "/home/duybao/PTW_AVISPA/AVISPA_2/AVIO_Inserts"
OUTPUT_PATH ="/home/duybao/PTW_AVISPA/AVISPA_2/AVIO_cropped"


from pathlib import Path
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

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

#Write cropped image to destination directory
def write_image(image, original_file,):
    # split file path by seperator "/" to get file name
    splits = str(original_file).split("/")
    #print("splits:", splits)
    filename = splits[-1]  #extract the filename at the end of directory: /home/filename.png
    new_filepath = OUTPUT_PATH + r"/" + filename
    print("new_filepath:",new_filepath)
    cv2.imwrite(new_filepath,image)

#creates a list containing the directories of all img-files ['home/image1.png', 'home/image2.png']
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

# Categorize images depending on camera angle. Depends on the filename(here: target_dir)
def getSub(target_dir:str) -> int :
    subNum = 0
    if '002' in target_dir:
        subNum = 2
    elif '005' in target_dir:
        subNum = 5
    elif '006' in target_dir:
        subNum = 6
    elif '007' in target_dir:
        subNum = 7
    elif '008' in target_dir:
        subNum = 8
    elif '009' in target_dir:
        subNum = 9
    elif '010' in target_dir:
        subNum = 10
    elif '011' in target_dir:
        subNum = 11
    elif '012' in target_dir:
        subNum = 12
    elif '013' in target_dir:
        subNum = 13

    return subNum

#categorizing some images into id-list
def getComponentID(target_path:str) -> str:
    id = ""
    if 'ref1' in target_path:
        id = 'ref1'
    return id

CONFIG={
    'ref1': {
        2: {"delta_x": 790, "delta_y": 680, "delta_h": 1650, "delta_w": 1650},
        5: {"delta_x": 790, "delta_y": 670, "delta_h": 1650, "delta_w": 1650},
        6: {"delta_x": 800, "delta_y": 830, "delta_h": 1650, "delta_w": 1650},
        7: {"delta_x": 850, "delta_y": 880, "delta_h": 1650, "delta_w": 1650},
        8: {"delta_x": 820, "delta_y": 840, "delta_h": 1650, "delta_w": 1650},
        9: {"delta_x": 840, "delta_y": 1000, "delta_h": 1650, "delta_w": 1650},
        10: {"delta_x": 840, "delta_y": 840, "delta_h": 1650, "delta_w": 1650},
        11: {"delta_x": 940, "delta_y": 830, "delta_h": 1650, "delta_w": 1650},
        12: {"delta_x": 950, "delta_y": 910, "delta_h": 1650, "delta_w": 1650},
        13: {"delta_x": 920, "delta_y": 810, "delta_h": 1650, "delta_w": 1650}
    },
}


if __name__=="__main__":
    file_list = get_files(FILE_PATH) #list containing the directories of all img-files

    for file_path in file_list:
        idf = getComponentID(file_path)
        subNum = getSub(file_path)
        cropped = crop(file_path, CONFIG[idf][subNum])
        write_image(cropped, file_path)

