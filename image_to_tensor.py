import os
import re
import numpy as np
import cv2
from PIL import Image

def get_labels(path_to_labels, to_round):
    with open(path_to_labels, "r") as file:
        labels = file.read().splitlines()

    if to_round:
        labels_int = [round(eval(label)) for label in labels]

    else:
        labels_int = [eval(label) for label in labels]

    return labels_int

def user_move_to_classes(path_to_make, path_to_folder, path_to_user_labels, path_to_user_animes):
    labels_int = get_labels(path_to_user_labels, to_round = False)
    list_frames = os.listdir(path_to_folder)

    path_to_anime_ids = "./anime-info/anime_ids.txt"
    with open(path_to_anime_ids, "r") as file:
        anime_ids = file.read().splitlines()

    with open(path_to_user_animes, "r") as file:
        user_animes = file.read().splitlines()

    for i, frame in enumerate(list_frames):
        current_frame = path_to_folder + frame
        rank_idx, frame_idx, *others = [int(occur_str) for occur_str in re.findall(r"\d+", frame)]
        frame_anime_id = anime_ids[rank_idx - 1] # 1 indexed

        try:
            user_anime_index = user_animes.index(frame_anime_id)

        except:
            pass

        else:
            frame_label = labels_int[user_anime_index]
            destination = path_to_make + str(frame_label)

            if not os.path.exists(destination):
                os.makedirs(destination)

            os.system(f"cp {current_frame} {destination}")

def move_to_classes(path_to_make, path_to_folder):
    labels_int = get_labels(path_to_labels, to_round = False)
    list_frames = os.listdir(path_to_folder)

    for i, frame in enumerate(list_frames):
        current_frame = path_to_folder + frame
        rank_idx, frame_idx, *others = [int(occur_str) for occur_str in re.findall(r"\d+", frame)]

        frame_label = labels_int[rank_idx - 1] # 1 indexed
        destination = path_to_make + str(frame_label)

        if not os.path.exists(destination):
            os.makedirs(destination)

        os.system(f"cp {current_frame} {destination}")

def add_noise(path_to_images, path_to_noised_images, std):
    list_images = os.listdir(path_to_images)

    if not os.path.exists(path_to_noised_images):
        os.makedirs(path_to_noised_images)

    for i, image in enumerate(list_images):
        current_image = cv2.imread(path_to_images + image)
        row, col, ch = current_image.shape
        gaussian = np.random.normal(mean, std, (row, col, ch))
        noisy_image = current_image - gaussian.astype(np.uint8)
        noisy_image = np.clip(noisy_image, 0, 255)

        image_name = image[0:-4]
        destination = path_to_noised_images + image_name + "_" + str(mean) + "_" + str(std) + ".jpg"
        cv2.imwrite(destination, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    args = sys.argv
    globals()[args[1]](*args[2:])