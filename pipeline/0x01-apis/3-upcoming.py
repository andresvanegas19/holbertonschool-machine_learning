#!/usr/bin/env python3
""" data collection from APIs """
from requests import get


if __name__ == '__main__':
    req_launches = get(
        "https://api.spacexdata.com/v4/launches/upcoming"
    )
    launches = sorted(req_launches.json(), key=lambda i: i['date_unix'])
    date_unix = launches[0]['date_unix']

    for i in req_launches.json():
        if i['date_unix'] == date_unix:
            launch_name = i['name']
            date = i['date_local']
            rocket_id = i['rocket']
            launchpad_id = i['launchpad']
            break

    req_rockets = get(
        "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id)
    )

    req_launchpads = get(
        "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id)
    )

    print("{} ({}) {} - {} ({})".format(
        launch_name,
        date,
        req_rockets.json()['name'],
        req_launchpads.json()['name'],
        req_launchpads.json()['locality'],)
    )
