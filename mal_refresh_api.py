import requests
from urllib.parse import urlencode

# app info
with open("client_id.txt", "r") as file:
    client_id = file.read().strip()

with open("refresh_token.txt", "r") as file:
    refresh_token = file.read().strip()

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
