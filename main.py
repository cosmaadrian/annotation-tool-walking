import argparse
import numpy as np
import yaml
import os
import cv2
import pandas as pd
import pickle
import json
import utils
import uuid

coco2openpose = np.array([
    [0, 0],
    [1, 14],
    [2, 15],
    [3, 16],
    [4, 17],
    [5, 2],
    [6, 5],
    [7, 3],
    [8, 6],
    [9, 4],
    [10, 7],
    [11, 8],
    [12, 11],
    [13, 9],
    [14, 12],
    [15, 10],
    [16, 13],
    [17, 1],
])

def normalise_data(poses):
    middle_hips = (poses[:, 12, :2] + poses[:, 11, :2]) / 2
    y_distance = np.sqrt(np.sum((poses[:, 0, :2] - middle_hips) ** 2, axis = 1))

    y_distance[y_distance == 0] = np.mean(y_distance)
    x_distance = y_distance / 2

    if np.any(np.isnan(x_distance)) or np.any(x_distance == 0):
        print(x_distance[x_distance == 0])
        print(x_distance)

    y_distance = y_distance.reshape((poses.shape[0], 1))
    x_distance = x_distance.reshape((poses.shape[0], 1))

    poses[:, :, :2] = poses[:, :, :2] - middle_hips.reshape((poses.shape[0], 1, 2))
    poses[:, :, 0] = poses[:, :, 0] / x_distance
    poses[:, :, 1] = poses[:, :, 1] / y_distance

    return poses

def annotate(poses):
    while True:
        for i in range(len(poses)):
            canvas = np.zeros((270, 480, 3))
            for idx, (x, y, _) in enumerate(poses[i]):
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow('winname', canvas)
            key = cv2.waitKey(24)
            if key != -1:
                if chr(key) == 'n':
                    return 'normal'

                if chr(key) == 'a':
                    return 'abnormal'

                if chr(key) == 's':
                    return 'skip'

parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('--config_file', type = str, default='config.yaml')
parser.add_argument('--video', type = str, default = 'test/manhattan_times-30.mkv')
# parser.add_argument('--video', type = str, default = 'test/village.mp4')
args = parser.parse_args()

cfg = yaml.load(open(f'{args.config_file}', 'rt'), Loader = yaml.FullLoader)
for key, value in cfg.items():
    args.__dict__[key] = value

os.makedirs(f'{args.output_folder}/normal', exist_ok=True)
os.makedirs(f'{args.output_folder}/abnormal', exist_ok=True)
video_basename = args.video.split('/')[-1]

with open(f'{args.alphapose_path}/{args.video}/tracks.pkl', 'rb') as f:
    data = pickle.load(f)

track_ids = np.array([d['track_id'] for d in data])
ids, counts = np.unique(track_ids, return_counts = True)

filtered_track_ids = ids[np.argwhere(counts >= args.period_length)]
np.random.shuffle(filtered_track_ids)

for idx, track_id in enumerate(filtered_track_ids):
    pose_idxs = np.argwhere(track_ids == track_id)
    poses = np.array([data[idx.ravel()[0]]['pose'] for idx in pose_idxs])
    keypoints = normalise_data(poses)

    sequence_start = np.random.randint(poses.shape[0] - args.period_length)
    pose_subset = poses[sequence_start: sequence_start + args.period_length]
    mask = np.int32(pose_subset[:, :, 2] > 0.5).reshape((-1, 17, 1))
    pose_subset = pose_subset * mask

    annotation_result = annotate(pose_subset * 100 + 100)
    if annotation_result not in ['normal', 'abnormal']:
        continue

    joint_17 = (pose_subset[:, 5, :] + pose_subset[:, 6, :]) / 2
    joint_17 = joint_17.reshape((-1, 1, 3))
    pose_subset = np.hstack((pose_subset, joint_17))
    pose_subset[:, :, 0][pose_subset[:, :, 2] == 0] = 0
    pose_subset[:, :, 1][pose_subset[:, :, 2] == 0] = 0
    pose_subset[:, coco2openpose[:, 0]] = pose_subset[:, coco2openpose[:, 1]]

    np.save(f'data/{annotation_result}/{uuid.uuid4()}', pose_subset)
