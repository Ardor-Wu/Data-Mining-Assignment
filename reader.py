import os
import numpy as np

def load(name):
    with open(name, "r") as file:
        data = file.read().split('\n')[6:-1]
        mode = name.split("_")[1][:-4]
        latitude = np.array([item.split(',')[0] for item in data])
        longitude = np.array([item.split(',')[1] for item in data])
        return latitude, longitude, mode


filenames = os.listdir("H:/下载/Geolife Trajectories 1.3/result/")
for filename in filenames:
    load("H:/下载/Geolife Trajectories 1.3/result/" + filename)
