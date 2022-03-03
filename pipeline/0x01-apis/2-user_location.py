#!/usr/bin/env python3
""" data collection from APIs """
import sys
# import requests
from requests import get
import time


if __name__ == '__main__':
    """ Script that prints the location of a specific user """

    # https://api.github.com/users/holbertonschool
    url = sys.argv[1]
    # user is passed as first argument of the script with the full API URL
    request = get(url, params={'Accept': "application/vnd.github.v3+json"})

    if request.status_code == 200:
        print(request.json()["location"])

    if request.status_code == 404:
        print("Not found")

    if request.status_code == 403:
        limit = request.headers["X-Ratelimit-Reset"]
        x = int((int(limit) - int(time.time())) / 60)
        print(f"Reset in {x} min")
