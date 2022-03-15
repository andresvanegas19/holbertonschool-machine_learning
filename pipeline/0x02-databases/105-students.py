#!/usr/bin/env python3
"""
function that returns all students sorted by average score
"""


def top_students(mongo_collection):
    """
    get all students sorted by average score
    Args:
        mongo_collection(pymongo.collection.Collection): will be the pymongo

    Returns:
        all students sorted by average score
    """

    stundent_result = []

    for student in mongo_collection.find():
        total_grades = 0

        for i, subject in enumerate(student["topics"]):
            total_grades = total_grades + subject["score"]

        student["averageScore"] = total_grades / (i + 1)
        stundent_result.append(student)

        # sort list
    r = sorted(
        stundent_result,
        key=lambda std: std["averageScore"],
        reverse=True
    )

    return r
