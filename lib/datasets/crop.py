from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import math
import logging
import numpy as np
from core.config import cfg
np.random.seed(666)
logger = logging.getLogger(__name__)


def crop_support(entry):
    # Get support box
    img_path = entry['image']
    all_box = entry['boxes']

    img = cv2.imread(img_path)
    if entry['flipped']:
        img = img[:, ::-1, :]
    img = img.astype(np.float32, copy=False)
    img -= cfg.PIXEL_MEANS
    img = img.transpose(2,0,1)
    data_height = int(img.shape[1])
    data_width = int(img.shape[2])
     
    all_box_num = all_box.shape[0]
    picked_box_id = np.random.choice(range(all_box_num))
    picked_box = all_box[picked_box_id,:][np.newaxis, :].astype(np.int16)
    
    x1 = picked_box[0][0]
    y1 = picked_box[0][1]
    x2 = picked_box[0][2] 
    y2 = picked_box[0][3]

    width = x2 - x1
    height = y2 - y1
    context_pixel = 16
    
    new_x1 = 0
    new_y1 = 0
    new_x2 = width
    new_y2 = height

    target_size = (320, 320) #(384, 384)   

    if width >= height:
        crop_x1 = x1 - context_pixel
        crop_x2 = x2 + context_pixel
   
        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + context_pixel
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width
            
        short_size = height
        long_size = crop_x2 - crop_x1
        y_center = int((y2+y1) / 2)
        crop_y1 = int(y_center - (long_size / 2))
        crop_y2 = int(y_center + (long_size / 2))
        
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + math.ceil((long_size - short_size) / 2)
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height

        crop_short_size = crop_y2 - crop_y1
        crop_long_size = crop_x2 - crop_x1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.float32)
        delta = int((crop_long_size - crop_short_size) / 2)
        square_y1 = delta
        square_y2 = delta + crop_short_size

        new_y1 = new_y1 + delta
        new_y2 = new_y2 + delta

        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]

        square[:, square_y1:square_y2, :] = crop_box
    else:
        crop_y1 = y1 - context_pixel
        crop_y2 = y2 + context_pixel
   
        # New_y1 and new_y2 will change when crop context or overflow
        new_y1 = new_y1 + context_pixel
        new_y2 = new_y1 + height
        if crop_y1 < 0:
            new_y1 = new_y1 + crop_y1
            new_y2 = new_y1 + height
            crop_y1 = 0
        if crop_y2 > data_height:
            crop_y2 = data_height
            
        short_size = width
        long_size = crop_y2 - crop_y1
        x_center = int((x2 + x1) / 2)
        crop_x1 = int(x_center - (long_size / 2))
        crop_x2 = int(x_center + (long_size / 2))

        # New_x1 and new_x2 will change when crop context or overflow
        new_x1 = new_x1 + math.ceil((long_size - short_size) / 2)
        new_x2 = new_x1 + width
        if crop_x1 < 0:
            new_x1 = new_x1 + crop_x1
            new_x2 = new_x1 + width
            crop_x1 = 0
        if crop_x2 > data_width:
            crop_x2 = data_width


        crop_short_size = crop_x2 - crop_x1
        crop_long_size = crop_y2 - crop_y1
        square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.float32)
        delta = int((crop_long_size - crop_short_size) / 2)
        square_x1 = delta
        square_x2 = delta + crop_short_size

        new_x1 = new_x1 + delta
        new_x2 = new_x2 + delta

        crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        square[:, :, square_x1:square_x2] = crop_box

    square = square.astype(np.float32, copy=False)
    square_scale = float(target_size[0]) / long_size
    square = square.transpose(1,2,0)
    square = cv2.resize(square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    square = square.transpose(2,0,1)

    new_x1 = int(new_x1 * square_scale)
    new_y1 = int(new_y1 * square_scale)
    new_x2 = int(new_x2 * square_scale)
    new_y2 = int(new_y2 * square_scale)

    # For test
    support_data = square
    support_box = np.array([[new_x1, new_y1, new_x2, new_y2]]).astype(np.float32)
    return support_data, support_box