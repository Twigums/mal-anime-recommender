import os
import requests

# create "images" directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

path_picture_urls = "./anime-info/top-tv/anime_picture_urls.txt"

with open(path_picture_urls, "r") as file:
    image_urls = file.read().splitlines()

for i, url in enumerate(image_urls, 1):
    response = requests.get(url)

    if response.status_code == 200:
        # Extract the file name from the URL
        filename = os.path.join("images/top-tv", f"image_{i}.jpg")

        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {filename}")

    else:
        print(f"Failed to download: {url}")

print("All images downloaded and saved in the 'images' folder.")
