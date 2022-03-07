#!/usr/bin/env python3
""" data collection from APIs """
from requests import get


def sentientPlanets():
    """
    method that returns the list of names of the home planets of all
    sentient species.
    """
    result_planets = []
    n_page = 1
    state = True

    while state:
        req_species = get(
            f"https://swapi-api.hbtn.io/api/species/?page=" + str(n_page)
        )
        data_species = req_species.json()

        for specie in data_species["results"]:
            if 'sentient' in {specie["classification"], specie['designation']}:
                if specie['homeworld'] is not None:
                    req_planet = get(specie['homeworld'])
                    result_planets.append(req_planet.json()['name'])

        if data_species['next'] is None:
            state = False

        n_page += 1

    return result_planets
