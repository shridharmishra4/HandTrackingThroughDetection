import math
import os

import cv2
import numpy as np
import skimage
from tqdm import tqdm
from mrcnn import model as modellib
from mrcnn.config import Config
import tensorflow as tf
import colorsys
import random
import scipy.io

import pickle


from keras import backend as K

from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0
#
# # On CPU/GPU placement
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)


def color_white(image, mask, angle):

    h, w, _ = image.shape
    alpha = 0.5
    color = [0, 255, 0]
    color_mask = np.zeros((h, w, 3))
    color_mask[:, :, :] = color
    color_mask = image * (1-alpha) + alpha * color_mask

    if mask.shape[-1] > 0:
        one_mask = (np.sum(mask, -1, keepdims=True) >= 1)
        colored = np.where(one_mask, color_mask, image).astype(np.uint8)
        
    else:
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        colored = gray.astype(np.uint8)

    _, _, num = mask.shape
    centroids = []
    
    for i in range(num):
        yy, xx = np.where(mask[:,:,i])
        dw = max(xx) - min(xx)
        dh = max(yy) - min(yy)
        dr = np.sqrt(dw**2 + dh**2)
        xc = int(np.mean(xx))
        yc = int(np.mean(yy))
        # print((yc,xc))
        # centroids.append([(yc,xc)])
        xe = int(xc + dr/2 * math.cos(angle[i]))
        ye = int(yc + dr/2 * math.sin(angle[i]))
        colored = cv2.line(colored, (xc, yc), (xe, ye), color=[255,0,0], thickness=2)
        colored[yc-1:yc+1,xc-1:xc+1,:] = [255, 255, 255]
        xe_a1 = int(xe - 5*math.cos(angle[i] + math.pi/6))
        ye_a1 = int(ye - 5*math.sin(angle[i] + math.pi/6))
        xe_a2 = int(xe - 5*math.cos(angle[i] - math.pi/6))
        ye_a2 = int(ye - 5*math.sin(angle[i] - math.pi/6))

        colored = cv2.line(colored, (xe, ye), (xe_a1,ye_a1), color=[255,0,0], thickness=2)
        colored = cv2.line(colored, (xe, ye), (xe_a2,ye_a2), color=[255,0,0], thickness=2)

    return colored, centroids

def load_model_custom(config):

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./model')
    model.load_weights('./model/trained_weights.h5', by_name=True)
    # print(model.summary())

    return model

class handConfig(Config):
    
    NAME = "hand"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  
    STEPS_PER_EPOCH = 1000
    # DETECTION_MIN_CONFIDENCE = 0.1
    # RPN_NMS_THRESHOLD = 0.1

    # DETECTION_NMS_THRESHOLD = 0.1
    # tf.global_variables_initializer()



class InferenceConfig(handConfig):
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.0001
    RPN_NMS_THRESHOLD = 0.0001
    # DETECTION_NMS_THRESHOLD = 0.1

# def random_colors(N, bright=True):
#
#     """
#     Generate random colors.
#     To get visually distinct colors, generate them in HSV space then
#     convert to RGB.
#     """
#     brightness = 1.0 if bright else 0.7
#     hsv = np.random.rand(N, 3)
#     hsv[:, 1] = 1
#     hsv[:, 2] = brightness
#     # hsv = [(i / N, 1, brightness) for i in range(N)]
#     colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
#     random.shuffle(colors)
#     return colors


# def vis_det_and_mask(im, class_name, dets, mask, thresh=0.8):
#     """Visual debugging of detections."""
#     num_dets = np.minimum(10, dets.shape[0])
#     colors_mask = random_colors(num_dets)
#     colors_bbox = np.round(np.random.rand(num_dets, 3) * 255)
#     # sort rois according to the coordinates, draw upper bbox first
#     draw_mask = np.zeros(im.shape[:2], dtype=np.uint8)
#
#     for i in range(1):
#         bbox = tuple(int(np.round(x)) for x in dets[i, :4])
#         # full_mask = unmold_mask(mask, bbox, im.shape)
#
#         score = dets[i, -1]
#         if score > thresh:
#             word_width = len(class_name)
#             cv2.rectangle(im, bbox[0:2], bbox[2:4], colors_bbox[i], 2)
#             cv2.rectangle(im, bbox[0:2], (bbox[0] + 18 + word_width*8, bbox[1]+15), colors_bbox[i], thickness=cv2.FILLED)
#             # apply_mask(im, full_mask, draw_mask, colors_mask[i], 0.5)
#             cv2.putText(im, '%s' % (class_name), (bbox[0]+5, bbox[1] + 12), cv2.FONT_HERSHEY_PLAIN,
# 								1.0, (255,255,255), thickness=1)
#     return im


def detect(model, img_dir):
    
    f = open("results.txt" , "w")

    for file_name in os.listdir(img_dir)[:2]:
        img_path = img_dir + file_name
        print("Processing image: ", file_name)
        img_origin = skimage.io.imread(img_path)
        img = img_origin.copy()
        result = model.detect([img], verbose=0)[0]
        pred_masks = result["masks"]
        pred_orientations = result["orientations"]
        pred_bbox = result["rois"]
        # print(pred_bbox)
        for box in pred_bbox:
            line = file_name + ',' + str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + '\n'
            f.write(line)
        save_img = img_origin
        save_img = color_white(save_img, pred_masks, pred_orientations)
        scipy.io.imwrite('./outputs/result_' + os.path.basename(img_path), save_img)
        print("output saved\n")
    
    f.close()


def process_frame(model, img):
    # img = img_origin.copy()
    result = model.detect([img], verbose=0)[0]
    # feature = (model.run_graph(molded_images, [("feature", self.keras_model.get_layer(name="roi_align_mask").output)]))
    # print(type(feature["feature"]), np.shape(feature["feature"]))


    pred_masks = result["masks"]
    pred_orientations = result["orientations"]
    pred_bbox = result["rois"]
    # deep_feature = result["deep_feature"]
    deep_feature = None
    scores = result["scores"]
    b_box = result["ROI_BBOX"]
    print(len(pred_bbox))
    # pprint(pred_bbox,scores)
    # print(np.shape(deep_feature))
    # print(pred_bbox)
    #         line = file_name + ',' + str(box[0]) + ',' + str(box[1]) + ',' + str(box[2]) + ',' + str(box[3]) + '\n'
    #         f.write(line)
    save_img = img
    save_img , centroids = color_white(save_img, pred_masks, pred_orientations)
    # global_centroids.append(centroids)
    # for frames in global_centroids:
    #     for centroids in frames:
    #         try:
    #             print(centroids)
    #             save_img = cv2.circle(save_img, (centroids[0][0],centroids[1][0]), radius=3, color=(255,0,0), thickness=-1)
    #         except:
    #             pass
    for bbox in b_box:
        centerCoord = (bbox[1] + int((bbox[3] - bbox[1]) / 2), bbox[0] + int((bbox[2] - bbox[0]) / 2))
        # print(centerCoord)
        # save_img = cv2.circle(save_img, centerCoord, 4, (255, 0, 255), -1)
        save_img = cv2.rectangle(save_img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255))
        # save_img = cv2.rectangle(save_img,(box[1],box[0]),(box[3],box[2]),(0,0,255))

    return save_img, pred_bbox,deep_feature
def divide_chunks(array, n):

    # looping till length l
    for i in range(0, len(array), n):
        yield array[i:i + n]

def detect_vids(model,img_dir):
    # model = load_model_custom(InferenceConfig())
    cap = cv2.VideoCapture(img_dir)
    # print(cap)
    ims = []
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(out_height,out_width,fps)

    save_file = f'./outputs/{img_dir.strip().split("/")[-1]}'
    # print('Enter object name: hands (left, right), duplo (yellow, white, black, red, orange, blue)')
    text = "hand_multi_gpu"
    mp4_file = save_file.split('.mp4')[0] + '_' + text + '.mp4'
    text_file = f"{save_file.split('.mp4')[0]}_{text}_{out_width}_{out_height}.txt"

    out = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc(*'MP4V'),
                          fps, (out_width, out_height))

    while True:
        ret, frame = cap.read()



        if not ret:
            break

        ims.append(frame)
    f = 0
    start_track = True
    list_bboxes = []
    print(len(ims))


    deep_feature_all = []
    for im in tqdm(ims[:200]):
        # print(f)
        # im = ims[f]
        # im = cv2.resize(im, (512, 360))

        if start_track:  # tracking
            deep_feature = None

            im, pred_bbox,deep_feature = process_frame(model, im)

            # exit(0)
            # feature = intermediate_output = intermediate_layer_model.predict(im)

            # state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            # location = state['ploygon'].flatten()
            # mask = state['mask'] > state['p'].seg_thr
            #
            # x1, y1, x2, y2 = int(min(location[::2])), int(min(location[1::2])), int(max(location[::2])), int(
            #     max(location[1::2]))
            # list_bboxes[f] = (x1, y1, x2, y2)
            #
            # im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            # cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imwrite('./outputs/result_' + str(f) +".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR) )
            list_bboxes.append(pred_bbox)
            deep_feature_all.append(deep_feature)
            f += 1
            # with open('parrot.pkl', 'wb') as file_handle:
            #     pickle.dump(deep_feature_all, file_handle)
            # print(list_bboxes)



        out.write(im)

    out.release()
    # with open('parrot.pkl', 'wb') as file_handle:
    #     pickle.dump(deep_feature_all, file_handle)
    #
    # with open('b_boxes.pkl', 'wb') as file_handle_2:
    #     pickle.dump(list_bboxes, file_handle_2)

if __name__ == '__main__':

    # import argparse
    # parser = argparse.ArgumentParser(description="Hand-Detector")
    # parser.add_argument("--image_dir", metavar="/path/to/directory/", help="Path to the image directory")
    # args = parser.parse_args()
    # img_dir = args.image_dir
    model = load_model_custom(InferenceConfig())
    # detect(model,"./frame_output/")
    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer("layer_name").output)
    detect_vids(model ,"../data_videos/short.mp4")