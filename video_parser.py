import os
import sys
import random
import youtube_dl
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import pytesseract
import numpy as np

# helper function: returns the path to the video after downloading to specified path
def download_video(url, name, path_output):
    ext = ".mp4"
    output = path_output + name + ext

    # params for youtube-dl
    ydl_opts = {
            "outtmpl": output,
            "format": "best[height<=480]/best",
            "geo_bypass": True,
            "geo_bypass_country": "JP",
            }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download = False)
        ydl.download([url])

    return output

# helper function: trim a video from the start to end percentages
# returns nothing
def trim_video(input_video, path_output, start, end):
    start = int(start)
    end = int(end)

    # params to find the length of a video since we can't splice through raw percentages
    duration_params = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format = duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_video
            ]

    duration = float(subprocess.check_output(duration_params).decode("utf-8").strip())
    start_time = str(duration * start / 100)
    end_time = str(duration * end / 100)

    # after calculating the times, these can be used for the params to trim the video
    params = [
            "ffmpeg", "-i", input_video,
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "copy", "-c:a", "copy",
            path_output
            ]

    subprocess.run(params, check = True)

# helper function: get random input number of frames and save to output path
def get_frames(path_video, path_output, frames):
    frames = int(frames)

    video_name = os.path.splitext(os.path.basename(path_video))[0]

    cap = cv2.VideoCapture(path_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = random.sample(range(frame_count), frames)

    # go over number of frames
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        # if frame does exist, then we should write it
        if success:
            frame_filename = os.path.join(path_output, f"{video_name}_{i}.jpg")
            cv2.imwrite(frame_filename, frame)

    cap.release()

# helper function: returns a boolean
def contains_japanese(text):

    # checks if character belongs in the japanese unicode family
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True

    return False


# threshold is given by the percentage; removes frames with a small range of variety
# returns nothing and saves videos to specified videos folder
def remove_single_color_frames(threshold):

    threshold = int(threshold)
    path_to_frames = "/mnt/b/YouTubeDL/anime-segmentation/output/test_img/"
    list_frames = os.listdir(path_to_frames)

    for frame in list_frames:
        current_frame = path_to_frames + frame

        image = cv2.imread(current_frame)
        std = np.std(image)

        if std <= threshold:
            os.remove(current_frame)
            print("removed " + current_frame)

# removes frames with Japanese text in it (according to pytesseract)
# returns nothing and saves videos to specified videos folder
def remove_jpn_frames():

    # flaming background range in hsv colors
    # low_range_color = np.array([0, 50, 50])
    # top_range_color = np.array([0, 255, 255])

    path_to_frames = "/mnt/b/YouTubeDL/3035-videos/frames/"
    list_frames = os.listdir(path_to_frames)

    for frame in list_frames:
        current_frame = path_to_frames + frame
        print(current_frame)

        image = cv2.imread(current_frame)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, low_range_color, top_range_color)
        masked_image = cv2.bitwise_and(image, image, mask = mask)

        text_data_masked = pytesseract.image_to_string(masked_image, lang = "jpn")
        text_data_grey = pytesseract.image_to_string(grey_image, lang = "jpn")

        if contains_japanese(text_data_grey) or contains_japanese(text_data_masked):
            os.remove(current_frame)
            print("removed")

# function to try to download video using the url if it is available (some videos are geo-locked, privated, etc.)_
# returns nothing and saves video in defined videos folder
def get_videos():
    if not os.path.exists("videos"):
        os.makedirs("videos")

    with open("./anime-info/top-tv/videos.txt", "r") as file:
        video_links = file.read().splitlines()

        for i, url in enumerate(video_links):
            path_output = "./videos/"

            if url != "None":
                try:
                    output_video = download_video(url, str(i + 1), path_output)
                except:
                    print("Could not get video for anime #", i)

# trims all videos in listed directory to given percentages
# returns nothing, and saves the trimmed video in given location
def trim_all(start_percentage, end_percentage):
    path_to_videos = "/mnt/b/YouTubeDL/videos/"
    list_videos = os.listdir(path_to_videos)

    for video in list_videos:
        trim_video(path_to_videos + video, path_to_videos + "trimmed" + video, start_percentage, end_percentage)

# generate random frames using the video listed in the directory
# returns nothing and saves frames into the output path
def random_frames_all(frames):
    path_to_videos = "/mnt/b/YouTubeDL/3035-videos/trimmed/"
    path_to_output = "/mnt/b/YouTubeDL/3035-videos/frames/"
    list_videos = os.listdir(path_to_videos)

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    for video in list_videos:
        get_frames(path_to_videos + video, path_to_output, frames)

# removes frames that breaks given rules
# returns nothing and only permanently removes frames
def remove_bad_frames():
    threshold = 25 # in percent

    # flaming background range in hsv colors
    low_range_color = np.array([0, 50, 50])
    top_range_color = np.array([0, 255, 255])

    path_to_frames = "/mnt/b/YouTubeDL/3035-videos/frames/"
    list_frames = os.listdir(path_to_frames)

    for frame in list_frames:
        current_frame = path_to_frames + frame
        print(current_frame)

        image = cv2.imread(current_frame)
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_image, low_range_color, top_range_color)
        masked_image = cv2.bitwise_and(image, image, mask = mask)

        text_data_masked = pytesseract.image_to_string(masked_image, lang = "jpn")
        text_data_grey = pytesseract.image_to_string(grey_image, lang = "jpn")
        std = np.std(grey_image)

        if contains_japanese(text_data_grey) or contains_japanese(text_data_masked) or std <= threshold:
            print("removed")

            os.remove(current_frame)

if __name__ == "__main__":
    args = sys.argv
    globals()[args[1]](*args[2:])
