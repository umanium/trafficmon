from abc import ABCMeta

__author__ = 'Luqman'

"""
Author: Luqman A. M.
ObjectTracking.py
Object Tracking Algorithms abstract for optical flow calculation (for two objects)
"""


class ObjectTrackingAbstract(object):
    __metaclass__ = ABCMeta
    algorithm_name = ""

    def __init__(self, name):
        self.algorithm_name = name
        pass
