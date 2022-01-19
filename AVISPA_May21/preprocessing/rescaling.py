import cv2
import os

INPUT_DIR = r"D:\PycharmProjects\ImagePreprocessing\data\flaw_01_01"
OUTPUT_DIR = r"D:\PycharmProjects\ImagePreprocessing\data\flaw_01_01\rescaled"

def get_files(target_dir):
    """ Get a list of all filepath in a directory """
    item_list = os.listdir(target_dir)

    file_list = list()
    for item in item_list:
        item_dir = os.path.join(target_dir,item)
        if os.path.isdir(item_dir):
            file_list += get_files(item_dir)
        else:
            file_list.append(item_dir)
    return file_list

def rescale_image(file):
    img = cv2.imread(file)
    return cv2.resize(img, (299, 299))

def save_file(image, filename):
    cv2.imwrite(os.path.join(OUTPUT_DIR,filename), image)

filelist = get_files(INPUT_DIR)

if os.path.exists(OUTPUT_DIR) == False:
   os.mkdir(OUTPUT_DIR)

for file in filelist:
    rescaled_img = rescale_image(file)
    filename = file.split("\\")[5]
    save_file(rescaled_img,filename)