import argparse
import yaml
import os
import pandas as pd
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2


df = pd.read_csv('/media/ssd/gait-recognition/data/youtube-plain/better_poses/annotations.csv')
df = df.sample(frac = 1.)
print(df.head())

def visualize(poses, repeat = False):
    while True:
        for i in range(len(poses)):
            canvas = np.zeros((270, 480, 3))
            for idx, (x, y, _) in enumerate(poses[i]):
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow('winname', canvas)
            cv2.waitKey(12)

        if not repeat:
            break

PATH = '/media/ssd/gait-recognition/annotation-tool-walking/'

magnitudes = []
length = []
confidences = []

for i, row in tqdm.tqdm(df.iterrows()):
    file = row['file_name']
    image = np.load(file)
    image[:, :, 1] = - image[:, :, 1]

    relevant_x = image[:, [11, 14], 0]
    relevant_y = image[:, [11, 14], 1]

    relevant_x = np.apply_along_axis(smooth, 0, relevant_x)
    relevant_y = np.apply_along_axis(smooth, 0, relevant_y)

    grad_x = np.gradient(relevant_x, 2, axis = 0)
    grad_y = np.gradient(relevant_y, 2, axis = 0)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2).sum(axis = -1).mean()
    magnitudes.append(magnitude)
    length.append(image.shape[0])
    confidences.append(np.mean(image[:, :, 2]))

df['magnitude'] = magnitudes
df['length'] = length
df['confidence'] = confidences
df = df.sort_values(by = 'magnitude')
df.to_csv(f'{PATH}/annotations_magnitudes.csv', index = False)