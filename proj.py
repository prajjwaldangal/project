import cv2
# Nov - 19 --> ML project presentation  --> 100 images 4a & 4c --> (Mon)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
DATAPATH = "/Users/prajjwaldangal/Documents/cs/summer2018/algo/previous/data/"

# this function loads image and converts to canny
def foo(path):
    sys.stdout.write(path)
    sys.stdout.flush()
    sys.stdout.write("\n")
    img = cv2.imread(path)
    img = cv2.resize(img, (40, 40))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(img, 100, 200)

# this function uses mpimg to load an image
def canny2(path):
    img = mpimg.imread(path)
    

def plotting(root, n, hair_type="4a", dataset_type="train", segmented=True):
    plt.figure(1, (10, 10))
    for i in range(n):
        if segmented:
            path = os.path.join(root, dataset_type, hair_type, "{}-segmented".format(hair_type), str(i+1)+".png")
        else:
            path = os.path.join(root, dataset_type, hair_type, "{}-unsegmented".format(hair_type), str(i+1)+".png")
        canny = foo(path)
        ax = plt.subplot(2,5, i+1)
        ax.set_title(str(i+1)+".png")
        plt.imshow(canny)
    plt.suptitle("Plot of {}".format(hair_type + " ({})".format("segmented" if segmented else "unsegmented")))
    plt.show()

# this is for more than 10 pictures
# this function is for batch plotting
def plotting2(root, n, hair_type="4a", dataset_type="train", segmented=True):
    """
    :param root: root directory
    :param n: it's better if n is a multiple of 10
    :param hair_type: hair type
    :param dataset_type: whether it's train or test data
    :param segmented:
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
            canny = foo(path)
            ax = plt.subplot(2,5, j+1)
            ax.set_title(str(image)+".png")
            plt.imshow(canny, cmap='binary')
            plt.waitforbuttonpress(-1)

    plt.show()

plotting2(DATAPATH, 20, "4a", segmented=True)

# what I'd rather already have:
