#!/usr/bin/env python3
"""  function that returns the list of school having a specific topic """


def schools_by_topic(mongo_collection, topic):
    """
    returns the list of school having a specific topic

    Args:
        mongo_collection will be the pymongo collection object
        topic (string) will be topic searched

    Returns:
        a list of dictionaries
    """
    re_match = []
    results = mongo_collection.find(
        {
            "topics": {
                "$all": [topic]
            }
        }
    )

    for result in results:
        re_match.append(result)

    return re_match
