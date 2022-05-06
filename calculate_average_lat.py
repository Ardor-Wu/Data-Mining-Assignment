import numpy as np
import pandas as pd
import os
from tqdm import tqdm

data_dir = '../data'
train_dir = os.path.join(data_dir, 'train_dataset/train').replace('\\', '/')
average_lats = []
for i in tqdm(range(1, 18329 + 1)):
    data = pd.read_csv(os.path.join(train_dir, str(i) + '.csv').replace('\\', '/'))
    lat = np.array(data.iloc[:, 1])
    average_lats.append(np.average(lat))

print(np.average(average_lats))
