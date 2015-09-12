from abc import ABCMeta

__author__ = 'Luqman'

"""
Author: Luqman A. M.
PixelCleaningAbstract.py
Abstract class for pixel cleaning algorithms
"""


class PixelCleaningAbstract(object):
    __metaclass__ = ABCMeta
    algorithm_name = ""

    def __init__(self, name):
        self.algorithm_name = name
        pass