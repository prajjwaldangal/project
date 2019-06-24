"""
Due by March 26, 2019

Develop a module that implements a steganography algorithm for an image format of your
choice (such as BMP or JPEG using program of your choice (Python, MATLAB, C/C++ or
Java). Your module should be able to encrypt “a message” into a copy of the media format
and provide a decryption algorithm, as well.

Side note: This task will require a lot of research on your part. Be certain that you’re up to
the challenge!
"""


import cv2
import matplotlib.pyplot as plt
import random

class Steganography:
    #msg_bin = []

    def __init__(self, msg=""):
        self.msg = msg

    def dec_to_bin(self, N):
        sg = ""
        if N == 0:
            return '0'
        while N != 0:
            sg += str(N % 2)
            N   =  N // 2
        return sg[::-1]

    def get_msg_bit_pattern(self):
        msg_bit_pattern = [self.dec_to_bin(ord(self.msg[i])) for i in range(len(self.msg))]
        self.msg_bit_pattern = ''.join(msg_bit_pattern)
        print("The message in binary format: {0}, length: {1}\n".format(self.msg_bit_pattern, len(self.msg_bit_pattern)))

    def get_image(self, path):
        # we encode the bit pattern in self.msg_bit_pattern into the image starting at row 4.
        img = cv2.imread(path)
        if img == []:
            print("invalid path")
            return
        img = cv2.resize(img, (80, 80), interpolation = cv2.INTER_AREA)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stego = img.copy()
        return stego, img

    def _draw_horizontal(self, start, end):
        pass

    def _draw_vertical(self, img, origin):
        length = int(0.05 * len(img))
        for i in range(origin[0], length+origin[0]+1):
            img[i][origin[1]][0] = 0
            img[i][origin[1]][1] = 0
            img[i][origin[1]][2] = 0
        return img

    def make_number(self, img, n):

        origin_x = int(0.8 * len(img))
        origin_y = int(0.8 * len(img[0]))

        if n == 1:
            self._draw_vertical(img, (origin_x, origin_y))
            self._draw_vertical(img, (origin_x, origin_y + int(0.05 * len(img))))
            pass
        return img


MSG = ""
IMG = [] #cv2.imread("/Users/prajjwaldangal/Desktop/picture.jpg")
copy = []
# def start():
MSG = input("Enter the message you want to encode\n")
s = Steganography(MSG)
# initially I thought I would do this:
# 1. get the message input
# 2. convert it to ascii in binary, a stream of 0s and 1s
# 3. Convert each pixel starting from fourth row to low or high according to the result of (2)
# But my current implementation makes more sense.
s.get_msg_bit_pattern()
path = "/Users/prajjwaldangal/Desktop/invitation.jpg"
copy, IMG = s.get_image(path)

if not MSG == "नेपाली":
    img = IMG

else:
    stego = s.make_number(copy, 1)
    img = stego
    # title = "Namaste"

fig = plt.figure(figsize=(5,4))
plt.subplot(1,1,1)
plt.imshow(img)
plt.title("Thanks for coming to the conference")
plt.show()
