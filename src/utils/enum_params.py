"""
Master Thesis
and
Lab Visualisation & Medical Image Analysis SS2019

Institute of Computer Science II

Author: Christian Breiderhoff
2019-2020
"""

from enum import Enum


class ConvNetType(Enum):
    """
    cost functions
    """
    U_NET_2D = 1       # Choose to use Cross Entropy Cost (Segmentation)


class Cost(Enum):
    """
    cost functions
    """
    CROSS_ENTROPY = 1       # Choose to use Cross Entropy Cost (Segmentation)
    DICE_SOFT = 2           # Choose to use Dice Coefficient Cost (Segmentation) -DICE
    MSE = 3                 # Choose to use Mean Squared Error Cost (TV Training / Regression)
    TV = 4                  # tv loss
    DICE_LOG = 5            # -log(DICE)
    BATCH_DICE_SOFT = 6     # SOFT DICE treating bacth as 3D tensor
    BATCH_DICE_LOG = 7      # LOG DICE treating bacth as 3D tensor
    DICE_SOFT_CE = 8        # loss = cost_weight * dice_loss + (1-cost_weight) * cross_entropy_loss
    BATCH_DICE_SOFT_CE = 9  # loss = cost_weight * batch_dice_loss + (1-cost_weight) * cross_entropy_loss


class DataModes(Enum):
    """
    Oparation Modes for the Data generator
    """
    TRAINING = 1
    VALIDATION = 2
    TESTING = 3


class Subtumral_Modes(Enum):
    """
    Oparation Modes for the Data generator
    """
    COMPLETE = 1
    CORE = 2
    ENHANCING = 3
    ALL = 4


class TrainingModes(Enum):
    """
    training modes
    """
    TVFLOW_REGRESSION = 1
    TVFLOW_SEGMENTATION = 2
    SEGMENTATION = 3


class Optimizer(Enum):
    """
    training optimzer
    """
    ADAGRAD = 1         # Adagrad Optimizer
    ADAM = 2            # Adam Optimzer
    MOMENTUM = 3        # SDG with Momentum


class Activation_Func(Enum):
    """
    training optimzer
    """
    RELU = 1
    SIGMOID = 2
    NONE = 3


class RestoreMode(Enum):
    """
    training optimzer
    """
    COMPLETE_SESSION = 1    # complete session including hyper params
    ONLY_BASE_NET = 2       # only base net without out put cov and variables for chnaged from tv to segmenation
    COMPLETE_NET = 3


class TV_clustering_method(Enum):
    """
    modes for clustering tv smoothed images
    """
    STATIC_BINNING = 1
    STATIC_CLUSTERS = 2
    K_MEANS = 3
    MEAN_SHIFT = 4