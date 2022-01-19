import argparse
import os

a = argparse.ArgumentParser()
a.add_argument("--pathIn", help="path to images")
a.add_argument('--type', default='train', type=str)
args = a.parse_args()


def get_files():
    files = os.listdir(args.pathIn)
    if len(files) == 0:
        print("No images found")
        exit()

    return files


def main():
    files = get_files()
    content = ''
    count = 0
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(args.pathIn, file)
            content += image_path + '\n'
            count += 1

    output_filename = args.type + '.txt'
    with open(output_filename, 'w') as output:
        output.write(content)

    print('Wrote %d lines to %s' % (count, output_filename))


if __name__ == "__main__":
    main()