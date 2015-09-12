from abc import ABCMeta

__author__ = 'Luqman'


class BackgroundSubtractionAbstract(object):
    __metaclass__ = ABCMeta
    background_model = None
    algorithm_name = ""

    def __init__(self, name):
        self.background_model = None
        self.algorithm_name = name

    def apply(self, image):
        pass