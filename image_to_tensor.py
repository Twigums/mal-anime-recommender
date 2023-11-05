import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def get_labels(path_to_labels, to_round):
    with open(path_to_labels, "r") as file:
        labels = file.read().splitlines()

    if to_round == True:
        labels_int = [round(eval(label)) for label in labels]

    else:
        labels_int = [eval(label) for label in labels]

    return labels_int

def create_dataset(path_to_folder, path_to_labels, transform = None):
    labels = get_labels(path_to_labels, to_round = True)

    images = []
    list_frames = os.listdir(path_to_folder)

    for frame in list_frames:
        current_frame = path_to_folder + frame
        rank_idx, frame_idx = [int(occur_str) for occur_str in re.findall(r"\d+", frame)]

        frame_label = labels[rank_idx - 1] # 1 indexed

        image = Image.open(current_frame)

        if transform:
            image = transform(image)

        images.append((image, frame_label))

    return images

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
    ])

path_to_labels = "./anime-info/top-tv/anime_scores.txt"
path_to_folder = "/mnt/b/YouTubeDL/anime-segmentation/output/test_img/"

images = create_dataset(path_to_folder, path_to_labels, transform = transform)
print(images.shape)
