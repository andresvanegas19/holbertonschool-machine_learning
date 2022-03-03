#!/usr/bin/env python3
""" data collection from APIs """

from requests import get


if __name__ == '__main__':
    """ Script that displays the number of launches per rocket """

    launches_request = get(
        "https://api.spacexdata.com/v4/launches"
    )
    launches = {}

    for launch in launches_request.json():
        if launch['rocket'] not in launches:
            launches[launch['rocket']] = 1
        else:
            launches[launch['rocket']] += 1

    rockets_request = get(
        "https://api.spacexdata.com/v4/rockets/"
    )
    rockets = []

    for rocket in rockets_request.json():
        if rocket['id'] in launches:
            rockets.append(
                {
                    'rocket': rocket['name'],
                    'launches': launches[rocket['id']]
                }
            )
        else:
            continue

    launches = sorted(rockets, key=lambda i: i['rocket'])
    launches = sorted(rockets, key=lambda i: i['launches'], reverse=True)

    for launch in launches:
        print("{}: {}".format(launch['rocket'], launch['launches']))
