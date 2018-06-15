
from yolo import YOLO
from yolo import multi_tracking
import cv2
import sys
import time
import numpy as np
if __name__ == '__main__':
    video_path='as4.mp4'
    multi_tracking(YOLO(), video_path)
