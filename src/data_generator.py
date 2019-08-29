"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on August 2019
"""

from abc import abstractmethod
import src.utils.tf_utils as tf_utils
import configuration as config
import logging

class ImageDataGenerator:
    @abstractmethod
    def __init__(self):
        return

    def __exit__(self, exc_type, exc_value, traceback):
        self.data = None

    def __enter__(self):
        return self

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()

    @abstractmethod
    def _parse_function(self):
        raise NotImplementedError()