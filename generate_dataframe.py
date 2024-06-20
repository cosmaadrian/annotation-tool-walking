import pandas as pd
import glob
import os


data = {
    'file_name': [],
    'label': []
}

for image in glob.glob('data/**/*', recursive = True):
    if os.path.isdir(image):
        continue
    data['file_name'].append(os.path.abspath(image))
    data['label'].append(image.split('/')[1])

df = pd.DataFrame.from_dict(data)
df.to_csv('annotations.csv', index = False)
