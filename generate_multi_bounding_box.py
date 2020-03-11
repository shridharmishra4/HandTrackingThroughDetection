import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
import colorsys
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
import cv2
import pickle

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
class_names = ("BG","Hand")


os.environ["CUDA_VISIBLE_DEVICES"] = "10"  # Or 2, 3, etc. other than 0


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


config = InferenceConfig()


def load_model_custom(config):
    #     with tf.device(DEVICE):

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./model')
    model.load_weights('./model/trained_weights.h5', by_name=True)
    # print(model.summary())

    return model


model = load_model_custom(InferenceConfig())


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def generate_multiple_bbox(model, image):
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        #         ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        #         ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])

    Proposal_limit = 1000
    final_boxes_limit = 15

    proposals = utils.denorm_boxes(mrcnn['proposals'][0, :Proposal_limit], image.shape[:2])

    roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
    roi_class_names = np.array(class_names)[roi_class_ids]
    roi_positive_ixs = np.where(roi_class_ids > 0)[0]
    # How many ROIs vs empty rows?
    #     print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
    #     print("{} Positive ROIs".format(len(roi_positive_ixs)))

    keep = np.where(roi_class_ids > 0)[0]
    #     print("Keep {} detections:\n{}".format(keep.shape[0], keep))

    keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
    #     print("Remove boxes below {} confidence. Keep {}:\n{}".format(
    #         config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))

    ### Refinement

    roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]

    # Apply bounding box transformations
    # Shape: [N, (y1, x1, y2, x2)]
    refined_proposals = utils.apply_box_deltas(
        proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)

    ids = np.arange(0, len(keep) - 1)  # Display all
    # ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
    # ids = roi_positive_ixs[:-1]

    #     captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
    #                 for c, s in zip(roi_class_ids[keep][ids], roi_scores[keep][ids])]
    #     visualize.draw_boxes(image,
    #                          refined_boxes=refined_proposals[keep][ids],
    #                          captions=captions, title="ROIs After Refinement",
    #                          ax=get_ax())
    final_bboxes_before_NMS = []
    #     colors = random_colors(50)

    for bbox, score in zip(refined_proposals[keep][ids][:],
                           roi_scores[keep][ids][:]):
        #         centerCoord = (bbox[1] + int((bbox[3] - bbox[1]) / 2), bbox[0] + int((bbox[2] - bbox[0]) / 2))
        #         visualize.draw_box(bbox)
        #         color = [i * 255 for i in colors[random.randint(0,30)]]
        #         x1_y1_x2_y2_score = [bbox[1],bbox[0],bbox[3],bbox[2]]

        #         image = cv2.rectangle(image, (x1_y1_x2_y2_score[0], x1_y1_x2_y2_score[1]),
        #                               (x1_y1_x2_y2_score[2], x1_y1_x2_y2_score[3]), color)

        final_bboxes_before_NMS.append([bbox[1], bbox[0], bbox[3], bbox[2], score])

    #     for bbox in refined_proposals[keep][ids]:
    #         # print(centerCoord)
    #         # save_img = cv2.circle(save_img, centerCoord, 4, (255, 0, 255), -1)
    #         image = cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255))
    #         # save_img = cv2.rectangle(save_img,(box[1],box[0]),(box[3],box[2]),(0,0,255))

    return image, final_bboxes_before_NMS


from tqdm import tqdm


def detect_vids(model, video_path):
    # model = load_model_custom(InferenceConfig())
    cap = cv2.VideoCapture(video_path)
    # print(cap)
    ims = []
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(out_height,out_width,fps)

    save_file = f'./outputs/{video_path.strip().split("/")[-1]}'
    text = "MultiBBox"
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
    for im in tqdm(ims[:600]):
        if start_track:  # tracking

            im, pred_bbox = generate_multiple_bbox(model, im)
            #             print((pred_bbox))
            list_bboxes.append(pred_bbox)
            f += 1
        out.write(im)
    out.release()

    #     with open('parrot.pkl', 'wb') as file_handle:
    #         pickle.dump(deep_feature_all, file_handle)

    with open('b_boxes.pkl', 'wb') as file_handle_2:
        pickle.dump(list_bboxes, file_handle_2)

    return list_bboxes


list_bboxes = detect_vids(model ,"../data_videos/short.mp4")