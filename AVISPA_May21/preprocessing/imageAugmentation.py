from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import os
import argparse
import traceback
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True, help="Path to input data directory")
    parser.add_argument('--n-samples', type=int, required=True, help="Number of copies from an image")
    parser.add_argument('--out-dir', help="Path to output directory")
    return parser.parse_args()


def augment(filepath, out_dir, num_copies):
    # load the image
    img = load_img(filepath)
    baseName = str(filepath).split("\\")[-1].split(".")[0]

    # convert to numpy array
    data = img_to_array(img)

    # expand dimension to one sample
    sample = expand_dims(data, 0)

    # create image data augmentation generator
    aug = ImageDataGenerator(
                               rotation_range=30,
                               width_shift_range=0.05,
                               height_shift_range=0.05,
                               shear_range=0.01,
                               zoom_range=[0.9, 1.25],
                               horizontal_flip=False,
                               vertical_flip=False,
                               fill_mode='reflect',
                               data_format='channels_last',
                               brightness_range=[0.5, 1.5]
                             )

    gen = aug.flow(sample, batch_size=1)

    for i in range(num_copies):
        image = gen.next()
        imgName = baseName + '_'+ str(i) + '.jpg'
        im = Image.fromarray(image[0].astype('uint8'))
        im.save(os.path.join(out_dir, imgName))

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

if __name__=="__main__":
    try:
        args = parse_args()
        filelist = get_files(args.data_dir)
        for file in filelist:
            copy_img =augment(file,args.out_dir,args.n_samples)
    except Exception as e:
        print(e)
        traceback.print_exc()
