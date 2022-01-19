Betriebssystem:
---
Die untenstehenden Befehle sind unter Ubuntu 16.04 getestet. Das Programm läuft allerdings auch unter Windows.

Ubuntu Pakete
---
sudo apt-get install python3 git build-essential

Python Pakete
---
pip3 install keras==2.1.3 tensorflow-gpu==1.3.0 tensorflow==1.4.0 scikit-learn: 0.19.1 matplotlib seaborn scikit-image h5py

CUDA
---
CUDA 8.0: https://developer.nvidia.com/cuda-toolkit-archive
cuDNN 5.1: https://developer.nvidia.com/cudnn

Darknet (Nur für Object-Detection-YOLO-Netze benötigt)
---
- git clone https://github.com/pjreddie/darknet
- Wenn eine GPU genutzt werden soll: In darknet/Makefile GPU=1 und cuDNN=1
- Im Darknet-Ordner: make


Tensorflow Object Detection (optional, nur für Training der  Object-Detection-Netze in Tensorflow benötigt)
---
- git clone https://github.com/tensorflow/models.git
- Pfad in detect/scripts/train_tensorflow.sh anpassen
- Pfade in detect/models/tensorflow/*/pipeline.config anpassen