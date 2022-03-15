#!/usr/bin/env python3
""" Python function that lists all documents in a collection """


def list_all(mongo_collection):
    """
    lists all documents in a collection

    Args:
        mongo_collection(pymongo.collection.Collection): will be the pymongo
        collection object

    Returns:
        an empty list if no document in the collection
    """
    documents = []

    for doc in mongo_collection.find():
        documents.append(doc)

    return documents
