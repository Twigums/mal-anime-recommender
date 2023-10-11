import requests
from urllib.parse import urlencode
import secrets

def get_new_code_verifier() -> str:
    token = secrets.token_urlsafe(100)
    return token[:128]

code_verifier = code_challenge = get_new_code_verifier()

# app info
with open("client_id.txt", "r") as file:
    client_id = file.read().strip()  # Read and remove any leading/trailing whitespace

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
