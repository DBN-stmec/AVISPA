import sys, os
import cv2

if(len(sys.argv) <= 1):
  print('filename missing')
else:
  cwd = os.getcwd()
  filename = cwd + '/' + sys.argv[1]

  print(filename)
  if(os.path.isfile(filename)):
    vidcap = cv2.VideoCapture(filename)
    filename, file_extension = os.path.splitext(filename)
    print(filename + file_extension)
    success,image = vidcap.read()
    count = 0
    success = True

    if not os.path.exists(filename):
      os.makedirs(filename)

    while success:
      if(count % 10 == 0):
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite(filename + "/frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
