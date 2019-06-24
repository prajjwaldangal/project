# reorder into reorder_vector = [8 7 4 1 2 3 6 9];
def reorder(ls, order=[8, 7, 4, 1, 5, 2, 3, 6, 9]):
    store = [el for el in ls]
    for i in range(len(ls)):
        print(i, ls, order[i]-1)
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

def num(ls):
    bin = ""
    for idx, el in enumerate(ls):
        if idx == 4:
            continue
        s = str(el)
        if s != '0' and s != '1':
            return -1
        bin += s
    return int(bin, 2)

class LocalTernaryPatterns:
    def __init__(self, threshold=2):
        # store the number of points and radius
        # self.numPoints = numPoints
        # self.radius = radius
        self.threshold = threshold

    def describe(self, image, eps=1e-7):
        rows = len(image)
        cols = len(image[0])
        upper_mat_ltp = np.zeros((rows, cols))
        lower_mat_ltp = np.zeros((rows, cols))
        im = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_REFLECT)
        #eng = matlab.engine.start_matlab()
        #ltp_upper, ltp_lower = eng.LTP(image, 100)
        print(len(im), len(im[0]))
        vfunc = np.vectorize(foo)
        for row in range(1, rows+1):
            for col in range(1, cols+1):
                cent = im[row][col]
                pixels = []
                for i in range(row-1, row+2):
                    for j in range(col-1, col+2):
                        pixels.append(im[i][j])
                    #pixels.append(im[i][col-1:col+1+1])
                # say pixels = [-3,-2,3,2,1,2,4,3,4]
                print("Center: {}, Pixels: {}".format((row, col), pixels))
                low = cent - self.threshold # 1-2 = -1
                high = cent + self.threshold # 1+2 = 3
                out_ltp = vfunc(pixels, high, low) # [-1,3]-> [-1, -1, 0, 0, 0, 0, 1, 0, 1]

                upper_ltp = vfunc(out_ltp, 1, -1, "up")
                upper_ltp = reorder(upper_ltp)
                print (upper_ltp)
                upper_ltp = num(upper_ltp) # convert to dec representation
                upper_mat_ltp[row][col] = upper_ltp

                lower_ltp = vfunc(out_ltp, 1, -1, "low")
                lower_ltp = reorder(lower_ltp)
                lower_ltp = num(lower_ltp)
                lower_mat_ltp[row][col] = lower_ltp
        (hist, _) = np.histogram(upper_mat_ltp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        return upper_mat_ltp, lower_mat_ltp

mx2 = [max2([TranProb[i][TranList[i].index(j)] for i in range(n) if j in TranList[i]]) for j in range(n)]
