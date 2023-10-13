import requests
from urllib.parse import urlencode
import secrets
import sys
import time

# helper function: returns text
def get_text(path):
    with open(path, "r") as file:
        raw_text = file.read().strip()

    return raw_text 

# helper function: returns token
def get_new_code_verifier() -> str:
    token = secrets.token_urlsafe(100)

    return token[:128]

# helper function: requires limit, offset, and the auth token and returns anime ids as a list in the order of their current rank
def ranked_ids(limit, offset, access_token):
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

# helper function: requires anime_ids, fields, and access_token and returns anime details
def ranked_details(anime_id, fields, access_token):
    url = f"https://api.myanimelist.net/v2/anime/{anime_id}"

    params = {
            "fields": fields,
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

# helper writer function to write to a file
def writer(path, array, append_status):
    if append_status:
        write_type = "a"

    else:
        write_type = "w"

    with open(path, write_type) as file:
        for item in array:
            file.write(str(item) + "\n")

# returns nothing
def get_api():
    code_verifier = code_challenge = get_new_code_verifier()

    # app info
    path_client_id = "client_id.txt"
    client_id = get_text(path_client_id)

    print("Client ID:", client_id)

    redirect_uri = "https://github.com/Twigums/mal-anime-recommender"

    # obtain token by first authorizing user
    authorization_url = "https://myanimelist.net/v1/oauth2/authorize"
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "read_manga read_anime",
        "code_challenge": code_challenge,
        "code_challenge_method": "plain",
    }

    auth_url = authorization_url + "?" + urlencode(params)
    print(f"Visit this URL: {auth_url}")

    # apply auth code
    authorization_code = input("Enter the authorization code from the redirect URL: ")

    # exchange the authorization code for access token
    token_url = "https://myanimelist.net/v1/oauth2/token"
    token_data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": authorization_code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
    }

    # check response and print info
    response = requests.post(token_url, data = token_data)

    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info["access_token"]
        refresh_token = token_info["refresh_token"]

        with open("access_token.txt", "w") as file:
            file.write(access_token)

        with open("refresh_token.txt", "w") as file:
            file.write(refresh_token)

        print("Obtained token...")

    else:
        print(f"Token request failed with status code {response.status_code}: {response.text}")

# requires a client ID from get_api()
# returns nothing
def refresh_api():

    # app info
    path_client_id = "client_id.txt"
    path_refresh_token = "refresh_token.txt"
    client_id = get_text(path_client_id)
    refresh_token = get_text(path_refresh_token)

    print("Client ID:", client_id)

    # obtain token by first authorizing user
    token_url = "https://myanimelist.net/v1/oauth2/token"
    token_data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }

    response = requests.post(token_url, data = token_data)

    if response.status_code == 200:
        token_info = response.json()
        access_token = token_info["access_token"]
        refresh_token = token_info["refresh_token"]

        with open("access_token.txt", "w") as file:
            file.write(access_token)

        with open("refresh_token.txt", "w") as file:
            file.write(refresh_token)

        print("Successfully refreshed token...")

    else:
        print(f"Token request failed with status code {response.status_code}: {response.text}")

# requires token and returns nothing
# writes anime ids to file
def anime_ids(limit, init_offset, append_status):

    # correct types for the function
    limit = int(limit)
    init_offset = int(init_offset)
    append_status = bool(append_status)

    # app info
    path_access_token = "access_token.txt"
    access_token = get_text(path_access_token)

    print("Using token...\n")

    # api max is 500
    api_max = 500

    # empty list to join to
    anime_ids = []

    # loop over floor + 1 times
    for i in range((limit // api_max) + 1):
        offset = init_offset + api_max * i

        anime_ids += ranked_ids(min(limit, api_max), offset, access_token)
        limit = max(0, limit - api_max)

    path_ids = "./anime-info/anime_ids.txt"
    writer(path_ids, anime_ids, append_status)

# requires token and returns nothing
# writes anime details to respective files
def anime_details(append_status):

    fields = input("Available fields: id, title, main_picture, alternative_titles, start_date, end_date, synopsis, mean, rank, popularity, num_list_users, num_scoring_users, nsfw, genres, created_at, updated_at, media_type, status, my_list_status, num_episodes, start_season, broadcast, source, average_episode_duration, rating, studios, pictures, background, related_anime, related_manga, recommendations, statistics, videos. \n Please type wanted fields separated by commas: ")
    fields_list = fields.split(", ")

    # ask user for type of picture if they want pictures
    if "main_picture" in fields_list:
        size_type = input("I noticed you wanted 'main_picture'. Do you want 'medium' or 'large'?: ")

    # correct types for the function
    append_status = bool(append_status)

    # app info
    path_access_token = "access_token.txt"
    access_token = get_text(path_access_token)

    print("Using token...\n")

    # paths
    path_anime_ids = "./anime-info/anime_ids.txt"
    path_list = ["./anime-info/" + field + ".txt" for field in fields_list]

    # api max is 500
    api_max = 500

    # now we have ids, so we can loop over each to get info
    anime_dict = {field: [] for field in fields_list}

    with open(path_anime_ids, "r") as file:
        anime_ids = file.read().splitlines()

    for i, anime_id in enumerate(anime_ids):
        anime_details = ranked_details(anime_id, fields, access_token)

        for field in fields_list:

            if field == "main_picture":
                anime_dict[field].append(anime_details[field][size_type])

            elif field == "videos":
                if anime_details[field] ==  []:
                    anime_dict[field].append("None")

                else:
                    anime_dict[field].append(anime_details[field][0]["url"])

            else:
                anime_dict[field].append(anime_details[field])

        if i % 100 == 0:
            print("Finished with anime rank #" + str(i))
            time.sleep(60) # arbitrary number to bypass api call frequency

    for i, field in enumerate(fields_list):
        writer(path_list[i], anime_dict[field], append_status)

# requires token and returns nothing
# writes anime video links to a video file

if __name__ == "__main__":
    args = sys.argv
    globals()[args[1]](*args[2:])
