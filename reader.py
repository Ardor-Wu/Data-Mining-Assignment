import os
import numpy as np
from tqdm import tqdm
import scipy.io as scio


def load(name):
    with open(name, "r") as file:
        data = file.read().split('\n')[6:-1]
        mode = name.split("_")[1][:-4]
        latitude = np.array([item.split(',')[0] for item in data])
        longitude = np.array([item.split(',')[1] for item in data])
        return latitude, longitude, mode


def load_geolife():
    filenames = os.listdir("../data/Geolife_cleaned/")
    lats = []
    lons = []
    modes = []
    for filename in tqdm(filenames):
        data_item = load("../data/Geolife_cleaned/" + filename)
        lats.append(data_item[0])
        lons.append(data_item[1])
        modes.append(data_item[2])


def load_TRAFFIC(data_path):
    data = scio.loadmat(data_path)
    labels = data['truth']
    data = data['tracks_traffic']
    mode = []
    location = []
    for i in range(len(labels)):
        if labels[i][1] == 1:
            mode.append(labels[i][0])
            tem = data[i][0].T
            location.append(tem)
    return np.array(location), np.array(mode)
