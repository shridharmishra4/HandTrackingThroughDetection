import numpy as np
import argparse
import time
import cv2
from sort import *
from pprint import pprint
from tqdm import tqdm
import pickle

# create instance of SORT
mot_tracker = Sort()


with (open("./b_boxes.pkl", "rb")) as openfile:
    while True:
        try:
            frames = pickle.load(openfile)
        except EOFError:
            break


# print(len(frames))
def track_vids_with_sort(video_path):
    cap = cv2.VideoCapture(video_path)
    ims = []
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_file = f'./outputs/{video_path.strip().split("/")[-1]}'
    text = "tracker_sort"
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

    # x1 = bbox[1]
    # y1 = bbox[0]
    # x2 = bbox[3]
    # y2 = bbox[2]
    deep_feature_all = []
    for i, im in tqdm(enumerate(ims)):
        if start_track:  # tracking
            pred_bbox = []

            #             for items in tqdm(frames):
            #                 print(items)
            items = np.array(frames[i])
            trackers = mot_tracker.update(items)
            #                 print(trackers)
            for bbox in trackers:
                text = " {}".format(bbox[-1])
                #                     print(text, bbox)
                centroid = (bbox[0] + int((bbox[2] - bbox[0]) / 2), bbox[1] + int((bbox[3] - bbox[1]) / 2))
                cv2.putText(im, text, (int(centroid[0] - 20), int(centroid[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                #                     print([(bbox[0], bbox[1]), (bbox[2], bbox[3])])
                cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255))

        out.write(im)
    out.release()


track_vids_with_sort("../data_videos/short.mp4")