import pickle
import cv2
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse



# objects = []
# with (open("./b_boxes.pkl", "rb")) as openfile:
#     while True:
#         try:
#             objects.append(pickle.load(openfile))
#         except EOFError:
#             break
#
# all_centroid = []
# print(len(objects[0]))



# for frame in objects[0]:
#     centroid_per_frame = []
#     for bbox in frame:
#         centerCoord = (bbox[0] + (bbox[2] / 2), bbox[1] + (bbox[3] / 2))
#         centroid_per_frame.append(centerCoord)
#         plt.scatter(centerCoord[0], centerCoord[1])
#     all_centroid.append(centroid_per_frame)
#
# plt.show()
#
# pprint.pprint(all_centroid)


def paint_vids(img_dir):
    # model = load_model_custom(InferenceConfig())
    cap = cv2.VideoCapture(img_dir)
    # print(cap)
    ims = []
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # print(out_height,out_width,fps)

    save_file = f'./outputs/{img_dir.strip().split("/")[-1]}'
    text = "point_plot"
    mp4_file = save_file.split('.mp4')[0] + '_' + text + '.mp4'
    # text_file = f"{save_file.split('.mp4')[0]}_{text}_{out_width}_{out_height}.txt"

    out = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc(*'MP4V'),
                          fps, (out_width, out_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        ims.append(frame)
    print(len(ims))

    deep_feature_all = []
    for i in tqdm(range(len(ims))):
        # print(i)
        for bbox in (objects[0][i]):
            # print(bbox)
            centerCoord = (bbox[1]+int( (bbox[3]-bbox[1]) / 2),bbox[0]+int((bbox[2]-bbox[0]) / 2))
            # print(centerCoord)
            ims[i] = cv2.circle(ims[i],centerCoord,4,(255,0,255),-1)
            ims[i] = cv2.rectangle(ims[i], (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255))
            # exit(0)




        # print(f)
        # im = ims[f]
        # im = cv2.resize(im, (512, 360))

        out.write(ims[i])

    out.release()

#
# def track_hands(video):
#     cap = cv2.VideoCapture(video)
#     # print(cap)
#     ims = []
#     out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     # print(out_height,out_width,fps)
#
#     save_file = f'./outputs/{video.strip().split("/")[-1]}'
#     text = "point_plot"
#     mp4_file = save_file.split('.mp4')[0] + '_' + text + '.mp4'
#     # text_file = f"{save_file.split('.mp4')[0]}_{text}_{out_width}_{out_height}.txt"
#
#     out = cv2.VideoWriter(mp4_file, cv2.VideoWriter_fourcc(*'MP4V'),
#                           fps, (out_width, out_height))
#
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             break
#
#         ims.append(frame)
#     print(len(ims))
#
#     deep_feature_all = []
#     for i in tqdm(range(len(ims))):
#         # print(i)
#         for bbox in (objects[0][i]):

def save_frames(video):
    import cv2

    # Opens the Video file
    cap = cv2.VideoCapture(video)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        print(i)

        cv2.imwrite('./output/frame_output/Frame' + str(i) + '.jpg', frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

save_frames("/scratch/shri/Projects/Hand-CNN/outputs/short_tracker_sort.mp4")