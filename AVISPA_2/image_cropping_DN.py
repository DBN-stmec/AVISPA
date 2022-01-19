import cv2
import os

#How to use:
# 1. Enter your input folder-directory with the images in "FILE_PATH"
# 2. Enter your output folder-directory for the cropped images(299x299) in "OUTPUT_PATH"
# !!!! Make sure that your directory has this seperator: "/", because // or :// won't work. In that cas you must change the seperator in the fnc: "write_image()"

FILE_PATH = "/home/duybao/PTW_AVISPA/AVISPA_2/RNGN_12"
OUTPUT_PATH ="/home/duybao/PTW_AVISPA/AVISPA_2/comparecrop"


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

#write image to destination directory
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
    if '2_w' in target_dir:
        subNum = 2
    elif '4_w' in target_dir:
        subNum = 4
    elif '5_w' in target_dir:
        subNum = 5
    elif '6_w' in target_dir:
        subNum = 6
    elif '8_w' in target_dir:
        subNum = 8
    return subNum

#categorizing some images into id-list
def getComponentID(target_path:str) -> str:
    id = ""
    if 'RNGN19_CFAA0_2021' in target_path:
        id = 'RNGN19_CFAA0_2021'
    elif 'CNGA12_GMTK1_20211011' in target_path:
        id = 'CNGA12_GMTK1_20211011'
    elif 'RCGX12_CFAA0_2021' in target_path:
        id = 'RCGX12_CFAA0_2021'
    elif 'RNGN12_CFAA0_2021' in target_path:
        id = 'RNGN12_CFAA0_2021'
        #new
    elif 'RNGN12_GMTK1_2021' in target_path:
        id = 'RNGN12_GMTK1_2021'
    elif 'RNGN12_GMTK3_2021' in target_path:
        id = 'RNGN12_GMTK3_2021'
    elif 'RNGN12_GMTK4_2021' in target_path:
        id = 'RNGN12_GMTK4_2021'
    return id

CONFIG={
    'CNGA12_GMTK1_20211011':{
        2: {"delta_x": 520, "delta_y": 550, "delta_h": 1200, "delta_w": 1300},
        4: {"delta_x": 710, "delta_y": 500, "delta_h": 1100, "delta_w": 1200},
        5: {"delta_x": 530, "delta_y": 420, "delta_h": 1150, "delta_w": 1250},
        6: {"delta_x": 330, "delta_y": 340, "delta_h": 1100, "delta_w": 1250},
        8: {"delta_x": 600, "delta_y": 360, "delta_h": 1100, "delta_w": 1240},
    },
    'RCGX12_CFAA0_2021': {
        2: {"delta_x": 390, "delta_y": 800, "delta_h": 800, "delta_w": 1000},
        4: {"delta_x": 790, "delta_y": 750, "delta_h": 800, "delta_w": 1000},
        5: {"delta_x": 500, "delta_y": 650, "delta_h": 800, "delta_w": 1000},
        6: {"delta_x": 110, "delta_y": 520, "delta_h": 800, "delta_w": 1000},
        8: {"delta_x": 540, "delta_y": 550, "delta_h": 800, "delta_w": 1000},
    },
    'RNGN12_CFAA0_2021':{
        2: {"delta_x": 400, "delta_y": 680, "delta_h": 910, "delta_w": 950},
        4: {"delta_x": 780, "delta_y": 690, "delta_h": 910, "delta_w": 950},
        5: {"delta_x": 470, "delta_y": 600, "delta_h": 910, "delta_w": 950},
        6: {"delta_x": 300, "delta_y": 480, "delta_h": 910, "delta_w": 950},
        8: {"delta_x": 580, "delta_y": 400, "delta_h": 910, "delta_w": 950}
    },
    'RNGN12_GMTK1_2021': {#
        2: {"delta_x": 450, "delta_y": 450, "delta_h": 910, "delta_w": 950},
        4: {"delta_x": 780, "delta_y": 690, "delta_h": 910, "delta_w": 950},
        5: {"delta_x": 470, "delta_y": 600, "delta_h": 910, "delta_w": 950},
        6: {"delta_x": 300, "delta_y": 480, "delta_h": 910, "delta_w": 950},
        8: {"delta_x": 580, "delta_y": 400, "delta_h": 910, "delta_w": 950}
    },
    'RNGN12_GMTK3_2021': {#
        2: {"delta_x": 450, "delta_y": 500, "delta_h": 910, "delta_w": 950},
        4: {"delta_x": 900, "delta_y": 600, "delta_h": 910, "delta_w": 950},
        5: {"delta_x": 550, "delta_y": 500, "delta_h": 910, "delta_w": 950},
        6: {"delta_x": 400, "delta_y": 370, "delta_h": 910, "delta_w": 950},
        8: {"delta_x": 580, "delta_y": 400, "delta_h": 910, "delta_w": 950}
    },
    'RNGN12_GMTK4_2021': {#
        2: {"delta_x": 480, "delta_y": 550, "delta_h": 910, "delta_w": 950},
        4: {"delta_x": 850, "delta_y": 550, "delta_h": 910, "delta_w": 950},
        5: {"delta_x": 500, "delta_y": 400, "delta_h": 910, "delta_w": 950},
        6: {"delta_x": 350, "delta_y": 300, "delta_h": 910, "delta_w": 950},
        8: {"delta_x": 600, "delta_y": 400, "delta_h": 910, "delta_w": 950}
    },
    'RNGN19_CFAA0_2021': {
        2: {"delta_x": 600, "delta_y": 730, "delta_h": 1500, "delta_w": 1600},
        4: {"delta_x": 900, "delta_y": 750, "delta_h": 1500, "delta_w": 1600},
        5: {"delta_x": 650, "delta_y": 640, "delta_h": 1500, "delta_w": 1600},
        6: {"delta_x": 570, "delta_y": 650, "delta_h": 1600, "delta_w": 1500},
        8: {"delta_x": 730, "delta_y": 780, "delta_h": 1500, "delta_w": 1600}
    },
}


if __name__=="__main__":
    file_list = get_files(FILE_PATH) #list containing the directories of all img-files
    #print("FILEPATH:",FILE_PATH)
    #print("file_list:", file_list)

    for file_path in file_list:
        idf = getComponentID(file_path)
        subNum = getSub(file_path)
        #cropped = crop(file_path, CONFIG[idf][subNum])
        #write_image(cropped, file_path)
        if "RNGN12_GMTK1" in file_path:
            if "6_w" in file_path:
                cropped = crop(file_path, CONFIG[idf][subNum])
                write_image(cropped, file_path)

