#!/usr/bin/env python3
"""
adding the top 10 of the most present IPs in the collection nginx of
the database logs
"""
from pymongo import MongoClient


if __name__ == "__main__":

    method_list = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    db_address = 'mongodb://127.0.0.1:27017'

    # client.database.collection
    collection = MongoClient(db_address).logs.nginx
    print("{} logs".format(collection.count_documents({})))

    print("Methods:")
    for method in method_list:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    status_get = collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print("{} status check".format(status_get))

    load = [
        {"$sortByCount": '$ip'},
        {"$limit": 10},
        {"$sort": {"ip": -1}}
    ]

    print("IPs:")
    for ip in collection.aggregate(load):
        print("\t{}: {}".format(ip['_id'], ip['count']))
