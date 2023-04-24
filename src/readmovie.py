import cv2
import numpy as np
import tensorflow as tf
class ReadMovie:
    def __init__(self, path, size):
        self.path = path
        self.video = cv2.VideoCapture(path)
        self.flamecount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.WIDTH  = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.shape  = [size[0], size[1], 3] # 3は3チャネル
        self.size = size
    def readvideo(self):
        flames = np.empty(shape=(self.flamecount, self.size[0], self.size[1], 3))
        index = 0
        while True:
            ret, flame = self.video.read()
            if ret:
                flame = tf.image.resize(flame, (self.size[0], self.size[1]))
                flames[index] = flame
                index = index + 1
            else :
                break
        return flames