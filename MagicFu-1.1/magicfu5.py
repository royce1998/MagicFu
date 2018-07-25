#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""MagicFu-1.1: First stable version of MagicFu. Uses RaspberryPi
camera to capture an image every two seconds and process user
position and action. Current supported actions include resting in
a chair, exercising on a bicycle, and exercising with a yoga ball."""

__author__ = "Siyao Fu, Chuqiao Gu, Runzhuo Yang"
__credits__ = ["Siyao Fu", "Runzhuo Yang", "Chuqiao Gu", "Xiuzhong Wang"]
__version__ = 1.0
__status__ = "Prototype"

import argparse
import colorsys
import os
import random
import time # for debugging
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from ba import body_ratio, intersect_area, body_pos, isabove, isnext, isMoving, overall_area, distance_measurement, isfaraway
from yad2k.models.keras_yolo import yolo_eval, yolo_head
from picamera.array import PiRGBArray
from picamera import PiCamera
import io
import picamera
import subprocess
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #prevents warnings from printing

def distance(loc1, loc2):
    d1 = loc1[0] - loc2[0]
    d2 = loc1[1] - loc2[1]
    return (math.sqrt(d1 * d1 + d2 * d2))

def printpred(category):
    if category == 'chair':
        print('\n[INFO] Predicted scene: User is taking a rest!')
    elif category == 'bicycle':
        print('\n[INFO] Predicted scene: User is doing workout with Gym Bike!')
    elif category == 'yoga':
        print('\n[INFO] Predicted scene: User is doing workout with Yoga Ball!')

parser = argparse.ArgumentParser(
    description='IFA show demo - currently support up to three indoor scenes.')
parser.add_argument(
    '-m',
    '--model_path',
    help='path to h5 model file containing body of a YOLO_v2 model',
    default='model_data/tiny.h5')
parser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default='model_data/tiny_anchors.txt')
parser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to coco_classes.txt',
    default='model_data/coco_classes.txt')
parser.add_argument(
    '-d',
    '--debug',
    help='true for debugging mode',
    default='off'
    )
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .1',
    default=.1)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .3',
    default=.3)


def _main(args):

    model_path = os.path.expanduser(args.model_path)
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = os.path.expanduser(args.anchors_path)
    classes_path = os.path.expanduser(args.classes_path)
    debug = os.path.expanduser(args.debug)
    
    debugging = False
    if (debug != 'off'):
        debugging = True

    sess = K.get_session()

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags. Should be in model_data folder.'
    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)      # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)       # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))

    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)

    if debugging: print("Note that debugging mode will make processing time slower due to camera preview time.")
    cachedposition = (-1000, -1000)
    cachedprediction = ['nothing', 'nothing']

    with picamera.PiCamera(sensor_mode = 6) as camera:
        camera.resolution = (800, 640)
        #camera.video_stabilization = True
        #camera.shutter_speed = 800
        
        print("\nPreloading success! Now running...")
        while True:
            stream = io.BytesIO()
            
            if debugging:
                #camera.start_preview()
                #time.sleep(2)
                #camera.stop_preview()
                start_time = time.time()
            camera.capture(stream, 'jpeg')
            if debugging: print("Capture time : %.3f" % (time.time() - start_time))

            #stream.seek(0)
            if debugging: opening = time.time()
            image = Image.open(stream)
            if debugging: print("Open Image time : %.3f" % (time.time() - opening))

            if is_fixed_size:
                resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
            else:
                # Due to skip connection + max pooling in YOLO_v2, inputs must have
                # width and height as multiples of 32.
                new_image_size = (image.width - (image.width % 32),iZmage.height - (image.height % 32))
                resized_image = image.resize(new_image_size, Image.BICUBIC)
                image_data = np.array(resized_image, dtype='float32')
                print(image_data.shape)

            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
            if debugging: time1 = time.time()
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: image_data,
                    input_image_shape: [800, 640],
                    #input_image_shape: [640, 480],
                    K.learning_phase(): 0
                })

            if debugging: print("Sess Run time : %.3f"%(time.time()- time1))
            #font = ImageFont.truetype(
            #    font='font/FiraMono-Medium.otf',
            #    size=2 * np.floor(2e-2 * image.size[1] + 0.5).astype('int32'))
            #thickness = 2 * (image.size[0] + image.size[1]) // 300
            message = ''
            message_head = "\n[INFO] Predicted scene: User is"
            label_result = []
            match_pair = []
            dic = {'person': 'next to another person!', 'motorbike':'doing workout with a Gym Bicycle!', 'bicycle':'doing workout with a Gym Bicycle!', 'sports ball':'doing workout with a Yoga Ball!', 'apple':'doing workout with a Yoga Ball!', 'mouse':'doing workout with a Yoga Ball!', 'chair':'taking a rest!', 'sofa':'taking a rest!' }
            weights = {'apple':0.1, 'person': 0.7, 'bicycle': 0.1, 'sports ball': 0.1, 'motorbike': 0.1, 'mouse': 0.1, 'chair': 0.1, 'sofa':0.1}
            
            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]
                
                label = '{} {:.2f}'.format(predicted_class, score)
                label_result.append(predicted_class)
                
                #draw = ImageDraw.Draw(image)
                #label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                
                # w and h of the box
                w = bottom - top
                h = right - left
                x = top
                y = right
                # print(label, (left, top), (right, bottom))
                
                # add to match_pair for comparison
                if predicted_class in ['person','bicycle','sports ball', 'apple', 'motorbike', 'mouse', 'chair', 'sofa']: match_pair.append([c, [x, y, w, h]])
                
                # now the matched_pair contains only person and predefined items
                #if top - label_size[1] >= 0: text_origin = np.array([left, top - label_size[1]])
                #else: text_origin = np.array([left, top + 1])

                #if predicted_class in ['person']:
                #    for i in range(thickness):
                #        draw.rectangle(
                #            [left + i, top + i, right - i, bottom - i],
                #            outline=colors[c])
                message_origin = np.array([0, 10])

            person_box, ball_box, chair_box, bicycle_box = [], [], [],[]
            match_pair = sorted(match_pair)
            founduser = False
            for i in range(1,len(match_pair)):
                try:
                    test_pair = match_pair[i][0]
                    test_class = class_names[test_pair]
                    test_dic = dic[test_class]
                    if test_pair == match_pair[0][0]:
                        continue
                    
                    if intersect_area(match_pair[0][1],match_pair[i][1]) > 0:
                        if test_class in ['bicycle', 'motorbike']:
                            message = "Predicted scenery: Person is " + test_dic
                            print(message_head, test_dic, "\n[INFO] User Pos:", body_pos(match_pair[0][1]))
                            #draw.text(message_origin, message, fill=(255, 0, 0), font=font)
                            cachedprediction[0] = cachedprediction[1]
                            cachedprediction[1] = 'bicycle'
                            break
                        if test_class in ['sports ball', 'apple', 'mouse']:
                            message = "Predicted scenery: Person is " + test_dic
                            print(message_head, test_dic, "\n[INFO] User Pos:", body_pos(match_pair[0][1]))
                            #draw.text(message_origin, message, fill=(255, 0, 0), font=font)
                            cachedprediction[0] = cachedprediction[1]
                            cachedprediction[1] = 'yoga'
                            break
                        if test_class in ['chair', 'sofa']:
                            message = "Predicted scenery: Person is " + test_dic
                            print(message_head, test_dic, "\n[INFO] User Pos:", body_pos(match_pair[0][1]))
                            #draw.text(message_origin, message, fill=(255, 0, 0), font=font)
                            cachedprediction[0] = cachedprediction[1]
                            cachedprediction[1] = 'chair'
                            break
                    else:
                        founduser = True
                        if cachedprediction[1] != 'nothing': printpred(cachedprediction[1])
                        elif cachedprediction[0] != 'nothing': printpred(cachedprediction[0])
                        elif (distance(body_pos(match_pair[0][1]), cachedposition) < 7): print("\n[INF0] Predicted scene: User is taking a rest!")
                        print("[INFO] User Pos:", body_pos(match_pair[0][1]))
                        cachedposition = body_pos(match_pair[0][1])
                        cachedprediction[0] = cachedprediction[1]
                        cachedprediction[1] = 'nothing'
                        break
                    
                except KeyError:
                    continue
                
            if not founduser:
                cachedposition = (-1000, -1000)
            
            label_result = [] # cache
            if debugging: print("Loop time : %.3f\n" % (time.time() - start_time)) # -2 for the 2 second camera preview time
    
    sess.close()


if __name__ == '__main__':
    try:
        _main(parser.parse_args())
    except picamera.exc.PiCameraMMALError:
        subprocess.check_output(['pkill', '-9', '-f', '-o', 'magicfu.py'])
        print("\nRebooting camera. Please wait...")
        _main(parser.parse_args())
