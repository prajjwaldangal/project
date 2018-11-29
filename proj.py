import cv2
# Nov - 19 --> ML project presentation  --> 100 images 3c, 4a & 4c --> (Mon)
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


# DATAPATH = "/Users/prajjwaldangal/Documents/cs/summer2018/algo/proj/data/"
DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# this function loads image and converts to canny
def Bin(path):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (80, 80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV) # it's just plt that plots inv with black on hair part
    return img

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
            image = str(i*10+j+1)+".png"
            path = os.path.join(root, image)
            bin = Bin(path)
            cv2.imwrite(os.path.join(path, 'preprocessed_images', image), bin)
            ax = plt.subplot(2,5, j+1)
            ax.set_title(image)
            plt.imshow(bin, cmap='binary')
            # plt.waitforbuttonpress(-1)

    plt.show()

checkpoint_dir = "models/"

# /Users/prajjwaldangal/Documents/cs/summer2018/algo/previous/data/train/4a/4a-unsegmented/84.png

# just pass hair_type as argument
def rename(hair_type, segmented=False):
    if not segmented:
        path = os.path.join(DATAPATH, "train", hair_type, hair_type+"-unsegmented")
    else:
        path = os.path.join(DATAPATH, "train", hair_type, hair_type+"-segmented")
    filenames = os.listdir(path)
    try:
        filenames.remove('.DS_Store')
    except:
        pass
    try:
        filenames.remove('others')
    except:
        pass
    ls = [int(filename.split(".")[0]) for filename in filenames]
    len(filenames) == len(ls)
    ls.sort()
    for idx, filename in enumerate(ls):
        print("Converting {}.png to {}.png".format(filename, idx+1))
        os.rename(os.path.join(path, str(filename)+".png"), os.path.join(path, str(idx+1)+".png"))

from skimage import feature
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

# tensors= tuples in this case, (img, hair_type) high dimensional records returned by this function
def form_tensors(path, label=""):
    ret = []
    filenames = os.listdir(path)
    try:
        filenames.remove('.DS_Store')
    except:
        pass
    l = len(filenames)
    for i in range(1, l):
        image = os.path.join(path, str(i) + ".png")
        img = cv2.imread(image)
        img = cv2.resize(img, (80, 80))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if label:
            ret.append((gray, label))
        else:
            ret.append(gray)
    return ret

desc = LocalBinaryPatterns(24, 8)
def train(path):
    # tensors = image, hair_type,
    ls = []

    path_3c = os.path.join(path, "train", "3c", "3c-unsegmented")
    path_4a = os.path.join(path, "train", "4a", "4a-unsegmented")
    path_4c = os.path.join(path, "train", "4c", "4c-unsegmented")

    ls_3c = form_tensors(path_3c, "3c")
    # ls_4a = form_tensors(path_4a, "4a")
    ls_4c = form_tensors(path_4c, "4c")

    ls = ls_3c + ls_4c # + ls_4a

    ls = shuffle(ls)
    l = len(ls)

    data = []
    labels = []
    for i in range(l):
        hist = desc.describe(ls[i][0])
        labels.append(ls[i][1])
        data.append(hist)
    model = LinearSVC(C=100.0, random_state=42)
    model.fit(data, labels)
    return model

def test(model, path):
    path_3c = os.path.join(path, "train", "3c")
    path_4c = os.path.join(path, "test", "4c")
    ls_3c = form_tensors(path)
    ls_4c = form_tensors(path)
    ls = ls_3c + ls_4c
    ls = shuffle(ls)
    for img in ls:
        hist = desc.describe(img)
        # prediction = model.predict()

    

if __name__ == '__main__':
    # plotting2(DATAPATH, 20, "4a", segmented=True)
    # cap = cv2.VideoCapture(0)
    # print("Hello world")
    model = train(DATAPATH)
    test(model, DATAPATH)

    # test_model()fraud_detection
    # rename("4a")
    # train(DATAPATH)

# get video frame -> gray scale image -> resize into resized image (r_i) -> send in to NN, r_i acts the first layer
#               -> gives label at the end

# python project with LBP: https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
# LTP in matlab: https://stackoverflow.com/questions/27191047/calculating-the-local-ternary-pattern-of-an-image
