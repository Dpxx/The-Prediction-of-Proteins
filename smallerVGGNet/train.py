# USAGE
# python train.py --dataset train --model model_name.model --epochs 50

# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
# import random
# import pickle
import cv2
import os
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="train",
                help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", default="myModel.model",
                help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
ap.add_argument("-e", "--epochs", type=int, default=100,
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = args["epochs"]
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

csv_data = csv.reader(open("train.csv"))
dic = {}
for i in csv_data:
    dic[i[0]] = i[1]
dir_num = len(os.listdir("train"))
dir_list = os.listdir("train")

# initialize the trainX, testX, trainY and testY
# partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing]
trainX, testX, trainY, testY = [], [], [], []

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)

    # extract set of class labels from the image path and update the
    # labels list
    dir_name = imagePath.split(os.path.sep)[-2]
    label = dic[dir_name].split(";")
    l = np.zeros(shape=(10,))
    for i in label:
        l[int(i)] = 1

    if any(imagePath.__contains__(i) for i in dir_list[-int(dir_num * 0.2):]):
        testX.append(image)
        testY.append(l)
        continue
    trainX.append(image)
    trainY.append(l)
# scale the raw pixel intensities to the range [0, 1]
trainX = np.array(trainX, dtype="float") / 255.0
testX = np.array(testX, dtype="float") / 255.0
trainY = np.array(trainY)
testY = np.array(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}),
    finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
