import argparse
import torch
import yaml
import os
import pandas as pd
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import glob
from statsmodels.stats.stattools import durbin_watson
from scipy.signal import savgol_filter


def visualize(poses, repeat = False):
    while True:
        for i in range(len(poses)):
            canvas = np.zeros((550, 550, 3))
            for idx, (x, y, _) in enumerate(poses[i]):
                cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow('winname', canvas)
            cv2.waitKey(1)

        if not repeat:
            break

df = pd.read_csv('annotations_magnitudes.csv')
df = df[df['magnitude'] > 0.0035]
df = df[df['length'] < 250]
sns.relplot(x = 'magnitude', y = 'length', data = df)
plt.show()

sns.relplot(x = 'length', y = 'confidence', data = df)
plt.show()

for i, row in df.iterrows():

    image = np.load(row['file_name'])
    image[:, :, 1] = - image[:, :, 1]

    visualize(image * 100 + 188)