#!/usr/bin/env python3
""" data collection from APIs """
from requests import get


def availableShips(passengerCount):
    """
    a method that returns the list of ships that can hold a given number
    of passengers

    Args:
        passengerCount (int): the number of passengers

    Returns:
        If no ship available, return an empty list.
    """

    total_ships = []
    page = 1
    state = True

    while state:
        request = get(
            f"https://swapi-api.hbtn.io/api/starships/?page={page}"
        )
        data = request.json()

        for ship in data['results']:
            passenger = ship['passengers'].replace(',', "")

            if passenger.isnumeric() and int(passenger) >= passengerCount:
                total_ships.append(ship['name'])

        if data['next'] is None:
            state = False

        page += 1

    return total_ships
