import os
import requests

# create "images" directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# path to picture urls
path_picture_urls = ""

with open(path_picture_urls, "r") as file:
    image_urls = file.read().splitlines()

# for each url, we should try to download the image
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
