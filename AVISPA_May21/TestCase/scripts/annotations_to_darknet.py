import argparse
import os
import xml.etree.ElementTree as ET

a = argparse.ArgumentParser()
a.add_argument("--pathIn", required=True, help="path to annotations")
a.add_argument("--pathOut", required=True)
args = a.parse_args()

def get_files():
    files = os.listdir(args.pathIn)
    if len(files) == 0:
        print("No test images found")
        exit()

    return files

def main():
    files = get_files()
    for file in files:
        if file.endswith(".xml"):
            annotations_file = os.path.join(args.pathIn, file)

            tree = ET.parse(annotations_file)
            xml = tree.getroot()
            class_name = str(0)
            image_width = float(xml.findtext('./size/width'))
            image_height = float(xml.findtext('./size/height'))
            xmin = float(xml.findtext('./object/bndbox/xmin'))
            ymin = float(xml.findtext('./object/bndbox/ymin'))
            xmax = float(xml.findtext('./object/bndbox/xmax'))
            ymax = float(xml.findtext('./object/bndbox/ymax'))

            width = (xmax - xmin) / image_width
            height = (ymax - ymin) / image_height
            x = ((xmin + xmax) / 2 - 1) / image_width
            y = ((ymin + ymax) / 2 - 1) / image_height

            content = '%s %f %f %f %f' % (class_name, x, y, width, height)
            output_filename = file.replace('.xml', '.txt')
            output_file = os.path.join(args.pathOut, output_filename)

            with open(output_file, 'w') as output:
                output.write(content)

            print('Wrote %s to %s' % (content, output_filename))



if __name__ == "__main__":
    main()