"""
Lab Visualisation & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019
"""

from enum import Enum


class Cost(Enum):
    """
    cost functions
    """
    CROSS_ENTROPY = 1       # Choose to use Cross Entropy Cost (Segmentation)
    DICE_COEFFICIENT = 2    # Choose to use Dice Coefficient Cost (Segmentation)
    MSE = 3                 # Choose to use Mean Squared Error Cost (TV Training / Regression)


class DataModes(Enum):
    """
    Oparation Modes for the Data generator
    """
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3


class TrainingModes(Enum):
    """
    training modes
    """
    TVFLOW = 1
    SEGMENTATION = 2


class Optimizer(Enum):
    """
    training optimzer
    """
    ADAGRAD = 1         # Adagrad Optimizer
    ADAM = 2            # Adam Optimzer
    MOMENTUM = 3        # SDG with Momentum