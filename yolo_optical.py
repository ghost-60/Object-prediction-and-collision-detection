
from yolo import YOLO
from yolo import optical_flow



if __name__ == '__main__':
    video_path='as4.mp4'
    optical_flow(YOLO(), video_path)
