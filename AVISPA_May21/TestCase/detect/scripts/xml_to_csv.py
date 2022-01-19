import os
import glob
import random
import pandas as pd
import xml.etree.ElementTree as ET

NEW_IMAGE_WIDTH=800


def xml_to_csv(path):
    xml_list = []
    files = glob.glob(path + '/*.xml')
    random.shuffle(files)
    for xml_file in files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            label = member[0].text
            width = int(root.find('size')[0].text)
            height = int(root.find('size')[1].text)

            x_min = int(member[4][0].text)
            y_min = int(member[4][1].text)
            x_max = int(member[4][2].text)
            y_max = int(member[4][3].text)

            if NEW_IMAGE_WIDTH is not None:
                x_min *= int(NEW_IMAGE_WIDTH / width)
                y_min *= int(NEW_IMAGE_WIDTH / width)
                x_max *= int(NEW_IMAGE_WIDTH / width)
                y_max *= int(NEW_IMAGE_WIDTH / width)

                width = NEW_IMAGE_WIDTH
                height *= int(NEW_IMAGE_WIDTH / width)

            value = (root.find('filename').text,
                     width,
                     height,
                     label,
                     x_min,
                     y_min,
                     x_max,
                     y_max
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    cwd = os.getcwd()
    dir_name = os.path.basename(os.path.normpath(cwd))
    image_path = os.path.join(cwd, "annotations")
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('%s_labels.csv' % dir_name, index=None)
    print('Successfully converted xml to csv.')


main()
