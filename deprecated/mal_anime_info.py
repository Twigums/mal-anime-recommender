import requests
import time

# functions to efficiently proccess >500 titles
def get_ranking(limit, offset):
    url = "https://api.myanimelist.net/v2/anime/ranking"

    params = {
            "ranking_type": "tv",
            "limit": limit,
            "offset": offset,
    }

    headers = {
            "Authorization": f"Bearer {access_token}",
    }

    response = requests.get(url, params = params, headers = headers)

    if response.status_code == 200:
        anime_data = response.json()

        anime_ids = [entry["node"]["id"] for entry in anime_data["data"][:]]

        return anime_ids

    else:
        print(f"Failed to retrieve top anime. Status code: {response.status_code}")

        return []

def get_details(anime_id):
    url = f"https://api.myanimelist.net/v2/anime/{anime_id}"

    params = {
            "fields": "title, main_picture, synopsis, mean, rank",
    }

    headers = {
            "Authorization": f"Bearer {access_token}",
    }

    response = requests.get(url, params = params, headers = headers)

    if response.status_code == 200:
        anime_details = response.json()

        return anime_details

    else:
        print(f"Failed to retrieve information. Status code: {response.status_code}")

def writer(path, array, append):
    if append:
        write_type = "a"

    else:
        write_type = "w"

    with open(path, write_type) as file:
        for item in array:
            file.write(str(item) + "\n")

# app info
with open("access_token.txt", "r") as file:
    access_token = file.read().strip()

print("Using token...\n")
limit = int(input("Enter the number of top animes: "))
init_offset = int(input("What should the offset be? (0 for top of the list): "))
append = input("Input boolean for append to existing files: ")

# api max is 500
api_max = 500

# empty list to join to
anime_ids = []

# loop over floor + 1 times
for i in range((limit // api_max) + 1):
    offset = init_offset + api_max * i

    anime_ids += get_ranking(min(limit, api_max), offset)
    limit = max(0, limit - api_max)

# now we have ids, so we can loop over each to get info
anime_titles = []
anime_picture_urls = []
anime_synopses = []
anime_scores = []
anime_ranks = []

for i, anime_id in enumerate(anime_ids):
    anime_details = get_details(anime_id)

    anime_titles.append(anime_details["title"])
    anime_picture_urls.append(anime_details["main_picture"]["medium"])
    anime_synopses.append(anime_details["synopsis"])
    anime_scores.append(anime_details["mean"])
    anime_ranks.append(anime_details["rank"])
    print(f"Finished retrieving details for {anime_id}.")

    if i % 100 == 0:
        time.sleep(45) # arbitrary number to get around max api call frequency

path_ids = "./anime-info/anime_ids.txt"
path_titles = "./anime-info/anime_titles.txt"
path_picture_urls = "./anime-info/anime_picture_urls.txt"
path_synopses = "./anime-info/anime_synopses.txt"
path_scores = "./anime-info/anime_scores.txt"
path_ranks = "./anime-info/anime_ranks.txt"


writer(path_ids, anime_ids, append)
writer(path_titles, anime_titles, append)
writer(path_picture_urls, anime_picture_urls, append)
writer(path_synopses, anime_synopses, append)
writer(path_scores, anime_scores, append)
writer(path_ranks, anime_ranks, append)
