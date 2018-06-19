#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
import cv2
import colorsys
import os
import random
import math
import argparse
import re
import time
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from matplotlib import cm
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *

multi_tracking_enable = False
optical_flow_enable = False
arima_predict = True
depthMapEstimation = True
percentResize = 0.7

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5' # model path or trained weights path
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        boxes = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            if((not multi_tracking_enable) or (predicted_class == 'person' or predicted_class == 'car')):
                boxes.append((left, top, right, bottom))
            #print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

        end = timer()
        print(end - start)
        return image, boxes

    def close_session(self):
        self.sess.close()
        del self

color = np.random.randint(0,255,(100,3))



# Depth map init ===============================================================
def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


model_path = '/home/ghost/Desktop/Algos/monodepth-master/models/model_kitti'

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', default = model_path)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)
args = parser.parse_args()

params = monodepth_parameters(
    encoder=args.encoder,
    height=args.input_height,
    width=args.input_width,
    batch_size=2,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode="border",
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False)

left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
depth_model = MonodepthModel(params, "test", left, None)
# SESSION
config = tf.ConfigProto(allow_soft_placement=True)
depth_sess = tf.Session(config=config)

# SAVER
train_saver = tf.train.Saver()

# INIT
depth_sess.run(tf.global_variables_initializer())
depth_sess.run(tf.local_variables_initializer())
coordinator = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=depth_sess, coord=coordinator)

# RESTORE
restore_path = args.checkpoint_path.split(".")[0]
train_saver.restore(depth_sess, restore_path)
# ==============================================================================

def detect_video(yolo, video_path):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    #yolo.close_session()
    #Optical flow initilize

    return_value, old_frame = vid.read()
    if(optical_flow_enable):
        old_frame = cv2.resize(old_frame, (0,0), fx=0.4, fy=0.4)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    arrows = np.zeros_like(old_frame)
    #Optical flow end
    frameCount = 1
    startFrame = 10
    #bbox = (287, 23, 86, 320)
    boxes = []
    while True:
        return_value, frame = vid.read()
        frameCount += 1
        if(optical_flow_enable):
            frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
        image = Image.fromarray(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image, boxes = yolo.detect_image(image)
        img = frame.copy()
        #(boxes)
        #Optical flow start=====================================================

        # calculate optical flow
        if(optical_flow_enable):
            flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            arrows = np.zeros_like(frame)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            i = 0
            mag_norm = cv2.normalize(mag, None, 0, 10, cv2.NORM_MINMAX)
            # draw the tracks
            """for x in range(mag.shape[1]):
                for y in range(mag.shape[0]):
                    ang_norm = ang[y][x] * 180 / (np.pi)

                    x2 = int(x + mag_norm[y][x] * math.cos(ang_norm))
                    y2 = int(y + mag_norm[y][x] * math.sin(ang_norm))
                    if(i % 50 == 0):
                        arrows = cv2.arrowedLine(arrows, (x, y), (x2, y2), color[1].tolist(), 1)
                    i += 1
            img = cv2.add(frame,arrows)"""
            img = np.asarray(img)
            img, arrows = cropped_img(img, boxes, mag_norm, ang)
            img = cv2.add(frame,arrows)
        #cv2.imshow('frame',img)
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        #Optical flow end=======================================================

        #Yolo implementation ===================================================
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        if(optical_flow_enable):
            result = image_overlay(result, img, boxes)
        cv2.imshow("result", result)
        #Yolo implementation end ===============================================

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def multi_tracking(yolo, video_path):
    global percentResize
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    global multi_tracking_enable
    global arima_predict
    multi_tracking_enable = True
    tracker = cv2.MultiTracker_create()
    camera = cv2.VideoCapture(video_path)
    ok, image = camera.read()
    if not ok:
        print('Failed to read video')
        exit()
    boxes = []

    frameCount = 1
    startFrame = 1
    skipFrames = 25
    consecutiveframes = 1

    initialHistoryCount = 11
    skipHistory = 5

    extrapolate = 3
    dof = 0

    yoloCount = 0
    countCond = 0
    xhistory = []
    yhistory = []
    depth_history = []
    while(True):
        if(frameCount == startFrame):
            frame = Image.fromarray(image)
            frame, boxes = yolo.detect_image(frame)
            #yolo.close_session()
            break
        ok, image = camera.read()
        frameCount += 1
    #np.set_printoptions(suppress = True)
    boxes = np.asarray(boxes)
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    boxes = np.ndarray.tolist(boxes)
    prevBoxes = len(boxes)
    curBoxes = prevBoxes
    #print(boxes)
    #np.savetxt('boxes.txt', boxes, fmt = "%i")
    #return boxes
    #boxes = []
    manualBox = 0
    for i in range(manualBox):
        box = cv2.selectROI('tracking', image)
        boxes.append(box)

    for eachBox in boxes:
        eachBox = tuple(eachBox)
        xhistory.append([int(eachBox[0] + eachBox[2] / 2)])
        yhistory.append([int(eachBox[1] + eachBox[3] / 2)])
        ok = tracker.add(cv2.TrackerMedianFlow_create(), image, eachBox)

    while(True):
        ok, image=camera.read()
        if not ok:
            break
        orig_image = image.copy()

        if(prevBoxes != curBoxes):
            countCond += 1
        if(frameCount % skipFrames == 0):
            #print(consecutiveframes)
            consecutiveframes = 1
            tracker = cv2.MultiTracker_create()
            frame = Image.fromarray(image)
            boxes = []
            frame, boxes = yolo.detect_image(frame)
            yoloCount += 1
            boxes = np.asarray(boxes)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = np.ndarray.tolist(boxes)
            prevBoxes = len(boxes)
            curBoxes = None
            xhistory = []
            yhistory = []
            depth_history = []
            for eachBox in boxes:
                eachBox = tuple(eachBox)
                xhistory.append([int(eachBox[0] + eachBox[2] / 2)])
                yhistory.append([int(eachBox[1] + eachBox[3] / 2)])
                ok = tracker.add(cv2.TrackerMedianFlow_create(), image, eachBox)
            #frameCount += 1
            #continue

        ok, boxes = tracker.update(image)
        for i in range(len(boxes)):
            xhistory[i].append(int(boxes[i][0] + boxes[i][2] / 2))
            yhistory[i].append(int(boxes[i][1] + boxes[i][3] / 2))
        if(arima_predict and len(xhistory[0]) > initialHistoryCount):

            #if(len(xhistory[i]) > 27): dof = 5
            #print(xhistory[0])

            for i in range(len(boxes)):
                history = xhistory[i].copy()
                history = [xhistory[i][t] for t in range(0, len(xhistory[i]), skipHistory)]
                xmin = min(history)
                history[:] = [x - xmin for x in history]
                xmax = max(history)
                if(xmax == 0): xmax = 1
                history[:] = [x / xmax for x in history]
                #print('xh', len(history))
                for j in range(extrapolate):
                    xmodel = ARIMA(history, order = (dof, 1, 0))
                    xmodel_fit = xmodel.fit(disp = 0, maxiter=200)
                    xoutput = xmodel_fit.forecast()
                    history.append(xoutput[0])
                xhat = int((xoutput[0] * xmax) + xmin)
                #xhat = xoutput[0]
                history = yhistory[i].copy()
                history = [yhistory[i][t] for t in range(0, len(yhistory[i]), skipHistory)]
                ymin = min(history)
                history[:] = [y - ymin for y in history]
                #history = [yhistory[i][0], yhistory[i][int(len(yhistory[i]) / 2)], yhistory[i][len(yhistory[i]) - 1]]
                ymax= max(history)
                if(ymax == 0): ymax = 1
                history[:] = [y / ymax for y in history]
                #print('yh', len(history))
                for j in range(extrapolate):
                    ymodel = ARIMA(history, order = (dof, 1, 0))
                    ymodel_fit = ymodel.fit(disp = 0, maxiter=200)
                    youtput = ymodel_fit.forecast()
                    history.append(youtput[0])
                yhat = int((youtput[0] * ymax) + ymin)
                #yhat = youtput[0]
                cp1 = int(boxes[i][0] + boxes[i][2] / 2)
                cp2 = int(boxes[i][1] + boxes[i][3] / 2)
                cv2.arrowedLine(image, (int(xhistory[i][0]),int(yhistory[i][0])), (cp1, cp2), (0, 255, 0), 2)
                cv2.arrowedLine(image, (cp1, cp2), (xhat, yhat), (0, 0, 255), 2)
                #slope = math.abs(math.atan((yhat - cp2) / (xhat - cp1)))
                #speed = math.sqrt((yhat - cp2) * (yhat - cp2) + (xhat - cp1) * (xhat - cp1))
                #percentChange = 0.0
                #if(yhat >= cp2):

                p1 = (int(xhat - boxes[i][2] / 2), int(yhat - boxes[i][3] / 2))
                p2 = (int(xhat + boxes[i][2] / 2), int(yhat + boxes[i][3] / 2))
                cv2.rectangle(image, p1, p2, (255, 255, 255), 1)
        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, (200,0,0), 2)

        if(depthMapEstimation):
            depth_est = image_depth(orig_image)
            dof = 0
            current_depth_est = depth_est.copy()
            pred_depth_est = depth_est.copy()
            pd = 'OFF'
            for i in range(len(boxes)):
                p1 = (int(boxes[i][0]), int(boxes[i][1]))
                p2 = (int(boxes[i][0] + boxes[i][2]), int(boxes[i][1] + boxes[i][3]))
                current_depth = cal_depth_box(depth_est, p1, p2)
                if(len(depth_history) < len(boxes)):
                    depth_history.append([current_depth])
                else:
                    depth_history[i].append(current_depth)
                if(math.isnan(current_depth)):
                    continue
                if(len(depth_history[i]) > initialHistoryCount):
                    pd = 'ON'
                    history = depth_history[i].copy()
                    hisotry = np.nan_to_num(history)
                    history = [history[t] for t in range(0, len(history), skipHistory)]
                    dmin = min(history)
                    history[:] = [d - dmin for d in history]
                    dmax = max(history)
                    if(dmax == 0): dmax = 1
                    history[:] = [d / dmax for d in history]
                    for j in range(extrapolate):
                        dmodel = ARIMA(history, order = (0, 1, 0))
                        dmodel_fit = dmodel.fit(disp = 0, maxiter=200)
                        doutput = dmodel_fit.forecast()
                        history.append(doutput[0])
                    #print(doutput[0])
                    if(not math.isnan(doutput[0])):
                        dhat = int((doutput[0] * dmax) + dmin)
                    else:
                        dhat = current_depth

                    current_depth_est = set_depth(current_depth_est, p1, p2, current_depth)
                    if(math.isnan(current_depth)):
                        print("wtf just happened")
                    cv2.putText(current_depth_est,text=str(int(current_depth)), org=(p1[0], p1[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(0, 0, 255), thickness=1)
                    pred_depth_est = set_depth(pred_depth_est, p1, p2, dhat)
                    cv2.putText(pred_depth_est,text=str(int(dhat)), org=(p1[0], p1[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(0, 0, 255), thickness=1)

            cv2.putText(pred_depth_est, text=pd, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 0, 255), thickness=2)
            #cv2.namedWindow("curdepth", cv2.WINDOW_NORMAL)
            current_depth_est = cv2.resize(current_depth_est, (0,0), fx=percentResize, fy=percentResize)
            cv2.imshow('curdepth', current_depth_est)
            #cv2.namedWindow("predepth", cv2.WINDOW_NORMAL)
            pred_depth_est = cv2.resize(pred_depth_est, (0,0), fx=percentResize, fy=percentResize)
            cv2.imshow('predepth', pred_depth_est)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(image, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        image = cv2.resize(image, (0,0), fx=percentResize, fy=percentResize)
        cv2.imshow('tracking', image)
        frameCount += 1
        consecutiveframes += 1
        k = cv2.waitKey(1)
        if k == 27 : break # esc pressed
    print(yoloCount)
    print(countCond)
    yolo.close_session()

#Depth estimation exclusive ====================================================
def image_depth(image):
    global depth_sess
    global left
    global depth_model
    global config
    global coordinator
    global threads
    global restore_path
    input_image = np.asarray(image)
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    disp = depth_sess.run(depth_model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    result = Image.fromarray(np.uint8(cm.binary(disp_to_img)*255))
    result = result.convert("RGB")
    numpyResult = np.array(result)
    numpyResult = cv2.cvtColor(numpyResult, cv2.COLOR_RGB2GRAY)
    return numpyResult

def cal_depth_box(depth_est, p1, p2):
    box_depth = depth_est[p1[1]:p2[1], p1[0]:p2[0]]
    box_depth = np.nan_to_num(box_depth)
    return box_depth.mean()

def set_depth(depth_est, p1, p2, val):
    crop_im = depth_est[p1[1]:p2[1], p1[0]:p2[0]]
    crop_im.fill(val)
    depth_est[p1[1]:p2[1], p1[0]:p2[0]] = crop_im
    return depth_est
# ==============================================================================


#Optical flow exclusive functions ==============================================

def optical_flow(yolo, video_path):
    global optical_flow_enable
    optical_flow_enable = True
    detect_video(yolo, video_path)

def cropped_img(im, boxes, mag, ang):
    np.asarray(im)
    new_im = im.copy()
    new_im.fill(255)
    arrows = np.zeros_like(im)
    for i in range(len(boxes)):
        left, top, right, bottom = boxes[i]
        c = 1
        magn_avr = 0
        ang_avg = 0
        for x in range(left, right):
            for y in range(top, bottom):
                if(mag[y][x] < 0):
                    continue
                c += 1
                ang_norm = ang[y][x] * 180 / (np.pi)
                eqx = math.cos(ang_norm) * mag[y][x] + math.cos(ang_avg) * magn_avr
                eqy = math.sin(ang_norm) * mag[y][x] + math.sin(ang_avg) * magn_avr
                ang_avg = math.atan(eqy / eqx)
                magn_avr = math.sqrt(eqx * eqx + eqy * eqy)

        x = int((left + right) / 2)
        y = int((top + bottom) / 2)
        x2 = int(x + magn_avr * math.cos(ang_avg))
        y2 = int(y + magn_avr * math.sin(ang_avg))
        arrows = cv2.arrowedLine(arrows, (x, y), (x2, y2), color[1].tolist(), 1)
        crop_im = im[top:bottom, left:right]
        new_im[top:bottom, left:right] = crop_im
    return new_im, arrows

def image_overlay(im1, im2, boxes):
    im1.setflags(write=1)
    for i in range(len(boxes)):
        left, top, right, bottom = boxes[i]
        crop_im = im2[top:bottom, left:right]
        im1[top:bottom, left:right] = crop_im
    return im1
#===============================================================================


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, boxes = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


if __name__ == '__main__':
    detect_img(YOLO())
    tf.app.run()
