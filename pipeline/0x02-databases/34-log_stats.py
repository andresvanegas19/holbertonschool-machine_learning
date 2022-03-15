#!/usr/bin/env python3
""" script that provides some stats about Nginx logs stored in MongoDB """

from pymongo import MongoClient


if __name__ == "__main__":
    collection_logs = MongoClient('mongodb://127.0.0.1:27017').logs.nginx

    print("{} logs".format(collection_logs.count_documents({})))
    print("Methods:")

    protocol_methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    for method in protocol_methods:
        num_method = collection_logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_method))

    num_path = collection_logs.count_documents(
        {"method": "GET", "path": "/status"}
    )

    print("{} status check".format(num_path))
