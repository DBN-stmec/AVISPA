from helper import image as image_helper
import numpy as np
import cv2
from matplotlib import pyplot as plt

path = 'scripts/'

plt.figure(1)
#image_filenames = ["1.jpg", "2.jpg", "3.jpg"]
image_filenames = ["4.jpg"]
images = []
images_cs = []
images_eh = []
images_cl = []
target_size = 250
for i in range(0, len(image_filenames)):
    filename = image_filenames[i]
    image = cv2.imread(path + filename, 0)
    height, width = image.shape
    new_width, new_height = int(120/height * width), 120
    image = cv2.resize(image, (new_width, new_height))
    images.append(image)

    image_cs = image_helper.contrast_stretching(image)
    images_cs.append(image_cs)

    image_eh = cv2.equalizeHist(image)
    images_eh.append(image_eh)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    image_cl = clahe.apply(image)
    images_cl.append(image_cl)

for i in range(0, len(images)):
    image = images[i]
    plt.subplot(101 + 10 * len(images) + i)
    plt.xlabel('Grauwert')
    plt.ylabel('Häufigkeit')
    plt.hist(image.ravel(),256,[0,256], color=(185/255,15/255,34/255,1))

width = 6 * len(images)
height = 4

fig = plt.gcf()
fig.set_size_inches(width, height)
plt.savefig("histograms_raw.png")
#plt.show()
plt.gcf().clear()

for i in range(0, len(images)):
    image = images_cs[i]
    plt.subplot(101 + 10 * len(images) + i)
    plt.hist(image.ravel(),256,[0,256], color=(185/255,15/255,34/255,1))

fig = plt.gcf()
plt.xlabel('Grauwert')
plt.ylabel('Häufigkeit')
fig.set_size_inches(width, height)
plt.savefig("histograms_cs.png")
#plt.show()
plt.gcf().clear()

for i in range(0, len(images)):
    image = images_eh[i]
    plt.subplot(101 + 10 * len(images) + i)
    plt.hist(image.ravel(),256,[0,256], color=(185/255,15/255,34/255,1))

fig = plt.gcf()
plt.xlabel('Grauwert')
plt.ylabel('Häufigkeit')
fig.set_size_inches(width, height)

plt.savefig("histograms_eh.png")
#plt.show()
plt.gcf().clear()

for i in range(0, len(images)):
    image = images_cl[i]
    plt.subplot(101 + 10 * len(images) + i)
    plt.hist(image.ravel(),256,[0,256], color=(185/255,15/255,34/255,1))

fig = plt.gcf()
plt.xlabel('Grauwert')
plt.ylabel('Häufigkeit')
fig.set_size_inches(width, height)
plt.savefig("histograms_cl.png")
#plt.show()
plt.gcf().clear()

stacked = np.hstack(images)
cv2.imwrite(path + "stacked.jpg", stacked)
stacked_cs = np.hstack(images_cs)
cv2.imwrite(path + "cs.jpg", stacked_cs)
stacked_eh = np.hstack(images_eh)
cv2.imwrite(path + "eh.jpg", stacked_eh)
stacked_cl = np.hstack(images_cl)
cv2.imwrite(path + "cl.jpg", stacked_cl)