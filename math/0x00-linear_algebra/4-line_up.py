#!/usr/bin/env python3
""" This is a script that contain a function
That will add two arrays """


def add_arrays(arr1, arr2):
    """ adds two arrays element-wise """
    lenght = len(arr1)
    if lenght != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(lenght)]
