import os
import numpy as np
from tqdm import tqdm


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
