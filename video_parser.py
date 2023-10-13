import os
import sys
import youtube_dl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
import pytesseract
from PIL import Image

# returns the path to the video
def download_video(url, name, output_path):
    ext = ".mp4"
    output = output_path + name + ext

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

def trim_video(input_video, output_path, start, end):
    start = int(start)
    end = int(end)
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

    params = [
            "ffmpeg", "-i", input_video,
            "-ss", start_time,
            "-to", end_time,
            "-c:v", "copy", "-c:a", "copy",
            output_path
            ]

    subprocess.run(params, check = True)

def contains_japanese(text):

    # checks if character belongs in the japanese unicode family
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True

    return False


# image = Image.open("image.png")
# 
# text_data = pytesseract.image_to_string(image, lang = "jpn")
# 
# if contains_japanese(text_data):
#     print("hellosu")
# url = "https://www.youtube.com/watch?v=ofSxVONGKTU"
# name = "test"
# output_path = "./videos/"

# returns nothing and saves videos to specified videos folder
def get_videos():
    if not os.path.exists("videos"):
        os.makedirs("videos")

    with open("./anime-info/top-tv/videos.txt", "r") as file:
        video_links = file.read().splitlines()

        for i, url in enumerate(video_links):
            output_path = "./videos/"

            if url != "None":
                try:
                    output_video = download_video(url, str(i + 1), output_path)
                except:
                    print("Could not get video for anime #", i)

def trim_all(start_percentage, end_percentage):
    path_to_videos = "/mnt/b/YouTubeDL/videos/"
    list_videos = os.listdir(path_to_videos)

    for video in list_videos:
        trim_video(path_to_videos + video, path_to_videos + "trimmed" + video, start_percentage, end_percentage)

if __name__ == "__main__":
    args = sys.argv
    globals()[args[1]](*args[2:])
