import cv2
# Nov - 19 --> ML project presentation  --> 100 images 3c, 4a & 4c --> (Mon)
import os, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# python project with LBP: https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
#  # LTP in matlab: https://stackoverflow.com/questions/27191047/calculating-the-local-ternary-pattern-of-an-image


import argparse

from joblib import dump, load
from sklearn.metrics import precision_recall_curve, accuracy_score

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
def rename(hair_type, directory): #segmented=False, test=False):

    path = os.path.join(DATAPATH, directory)

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


# reorder into reorder_vector = [8 7 4 1 2 3 6 9];
def reorder(ls, order=[8, 7, 4, 1, 2, 3, 6, 9]):
    store = [el for el in ls]
    for i in range(len(ls)):
        store[i] = ls[order[i]-1]# -1 due to matlab index

    return store

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

# vector function: https://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html
def foo(a, high, low, cltr='out'):
    if cltr == 'out':
        if a < low:
            return -1
        elif a > high:
            return 1
        else:
            return 0
    elif cltr == 'up':
        if a == low:
            return 0
        else:
            return a
    else:
        if a == low:
            return 1
        else:
            return 0

# def out_part(ls, high, low):
#     ret = []
#     for num in ls:
#         if num < low:
#             ret.append(-1)
#         elif num > high:
#             ret.append(1)
#         else:
#             ret.append(0)
#     return ret
#
# def upper_part(ls):
#     ret = []
#     for num in ls:
#         if num == -1:
#             ret.append(0)
#         else:
#             ret.append(num)
#     return ret
#
# def lower_part(ls):
#     ret = []
#     for num in ls:
#         if num == -1:
#             ret.append(1)
#         else:
#             ret.append(0)
#
#     return ret

class LocalTernaryPatterns:
    def __init__(self, numPoints=24, radius=8, threshold=2):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
        self.threshold = threshold

    def describe(self, image, eps=1e-7):
        ltp_upper = []
        ltp_lower = []
        rows = 80
        cols = 80
        im = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_REFLECT)
        #eng = matlab.engine.start_matlab()
        #ltp_upper, ltp_lower = eng.LTP(image, 100)
        pixels = []
        vfunc = np.vectorize(foo)
        for row in range(1, rows+1):
            for col in range(1, cols+1):
                cent = im[row][col]
                for i in range(row-1, row+2):
                    for j in range(col-1, col+1+1):
                        pixels.append(im[i][j])
                    #pixels.append(im[i][col-1:col+1+1])
                # say pixels = [-3,-2,3,2,1,2,4,3,4]
                # print(pixels)
                low = cent - self.threshold # 1-2 = -1
                high = cent + self.threshold # 1+2 = 3
                out_ltp = vfunc(pixels, high, low) # [-1,3]-> [-1, -1, 0, 0, 0, 0, 1, 0, 1]

                upper_ltp = vfunc(pixels, 1, -1, "up")
                upper_ltp = reorder(upper_ltp)

                lower_ltp = vfunc(pixels, 1, -1, "low")
                lower_ltp = reorder(lower_ltp)

# tensors= tuples in this case, (img, hair_type) high dimensional records returned by this function
def form_tensor(path, label=""):
    ret = []
    filenames = os.listdir(path)
    try:
        filenames.remove('.DS_Store')
    except:
        pass
    l = len(filenames)
    for i in range(1, l+1):
        image = os.path.join(path, str(i) + ".png")
        print(image)
        img = cv2.imread(image)
        img = cv2.resize(img, (80, 80))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if label:
            ret.append((gray, label))
        else:
            ret.append(gray)
    return ret

lbp_desc = LocalBinaryPatterns(24, 8)
ltp_desc = LocalTernaryPatterns(24, 8)

FEATURE = ""
def train(path, learning_rate, feature='lbp'):
    # tensors = image, hair_type,
    global FEATURE = feature
    ls = []

    path_3c = os.path.join(path, "train", "3c", "3c-unsegmented")
    path_4a = os.path.join(path, "train", "4a", "4a-unsegmented")
    path_4c = os.path.join(path, "train", "4c", "4c-unsegmented")

    ls_3c = form_tensor(path_3c, "3c")
    # ls_4a = form_tensors(path_4a, "4a")
    ls_4c = form_tensor(path_4c, "4c")

    ls = ls_3c + ls_4c # + ls_4a

    ls = shuffle(ls)
    l = len(ls)

    data = []
    labels = []

    if feature.lower().strip() == 'lbp':
        desc = lbp_desc
    else
        desc = ltp_desc

    for i in range(l):
        hist = desc.describe(ls[i][0])
        labels.append(ls[i][1])
        data.append(hist)
    model = LinearSVC(C=learning_rate, random_state=42)
    model.fit(data, labels)
    dump(model, 'model.joblib')
    return model

def metrics(y_true, y_pred):
    p, r = precision_recall_curve(y_true, y_pred)
    return p, r, accuracy_score(y_true, y_pred)

def test(path):
    model = load('model.joblib')
    path_3c = os.path.join(path, "3c")
    path_4c = os.path.join(path, "4c")
    ls_3c = form_tensor(path_3c, "3c")
    ls_4c = form_tensor(path_4c, "4c")
    ls = ls_3c + ls_4c
    ls = shuffle(ls)
    correct = 0
    ys_pred = []
    ys_true = []

    if FEATURE = 'lbp':
        desc = lbp_desc
    else:
        desc = ltp_desc

    for img in ls:
        hist = desc.describe(img[0])
        prediction = model.predict(hist.reshape(1, -1))
        ys_pred.append(prediction)
        ys_true.append(img[1])
    #precision, recall, accuracy = metrics(ys_true, ys_pred)
    #print("\n\nMetrics:\nPrecision: {0}\nRecall: {1}\nAccuracy: {2}".format(precision, recall, accuracy))

if __name__ == '__main__':
    # plotting2(DATAPATH, 20, "4a", segmented=True)
    # rename("4c", dir="test/4c")
    model = train(DATAPATH, learning_rate = 333, 'ltp')
    test("/Users/prajjwaldangal/Documents/cs/summer2018/algo/proj/data/test")
    # rename("3c", dir="train/3c/3c-unsegmented")
    # train(DATAPATH)
