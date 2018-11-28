import cv2
# Nov - 19 --> ML project presentation  --> 100 images 4a & 4c --> (Mon)
import os, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--t', default='t',
                    help='test the program functionality or not')
# parser.add_argument()
args = parser.parse_args()


DATAPATH = "/Users/prajjwaldangal/Documents/cs/summer2018/algo/previous/data/"

# this function loads image and converts to canny
def canny(path):
    sys.stdout.write(path)
    sys.stdout.flush()
    sys.stdout.write("\n")
    img = cv2.imread(path)
    img = cv2.resize(img, (40, 40))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, 100, 200)

# this is for more than 10 pictures
# this function is for batch plotting
def plotting2(root, n, hair_type="4a", dataset_type="train", segmented=True):
    """
    :param root: root directory
    :param n: it's better if n is a multiple of 10
    :param hair_type: hair type
    :param dataset_type: whether it's train or test images
    :param segmented: load segmented or unsegmented images
    :return:
    """
    plt.figure(1, (10, 12))  
    l = int(n/10) + 1
    plt.suptitle("Plot of {}".format(hair_type + " ({})".format("segmented" if segmented else "unsegmented")))
    if segmented:
        root = os.path.join(root, dataset_type, hair_type, "{}-segmented".format(hair_type))
    else:
        root = os.path.join(root, dataset_type, hair_type, "{}-unsegmented".format(hair_type))
    for i in range(l):
        for j in range(10):
            image = i*10+j+1
            path = os.path.join(root, str(image)+".png")
            canny = canny(path)
            ax = plt.subplot(2,5, j+1)
            ax.set_title(str(image)+".png")
            plt.imshow(canny, cmap='binary')
            plt.waitforbuttonpress(-1)

    plt.show()

checkpoint_dir = "models/"

def train_model():
    # load a model if already there
    try:
        f = open('my.model', 'r')
        model = f.readlines()
    except:
    # else train a new model
        print("We will train a model")

def test_model():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    # plotting2(DATAPATH, 20, "4a", segmented=True)
    cap = cv2.VideoCapture(0)
    print("Hello world")
    train_model()
    # test_model()
    pass

# get video frame -> gray scale image -> resize into resized image (r_i) -> send in to NN, r_i acts the first layer
#               -> gives label at the end
