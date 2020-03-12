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
    BRATS_SEGMENTATION = 3


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


class Scores(Enum):
    LOSS = "Loss"
    CE = "CE"
    ERROR = "Err"
    ACC = "Acc"
    IOU = "IoU"
    DSC = "DSC"
    DSC_COMP = "DSC Comp."
    DSC_CORE = "DSC Core"
    DSC_EN = "DSC Enhan."
    DSCP = "DSCpP"
    DSCP_COMP = "DSCpP Comp."
    DSCP_CORE = "DSCpP Core"
    DSCP_EN = "DSCpP Enhan."
    IOUP = "IoUpP"
    LR = "LR"
    L1 = "L1"
    L2 = "L2"
    DLTL = "DLTL"
    DTL = "DTL"


ScoresLong = {
    Scores.LOSS: "Loss",
    Scores.CE: "Cross Entropy",
    Scores.ERROR: "Error",
    Scores.ACC: "Accuracy",
    Scores.IOU: "IoU",
    Scores.DSC: "Dice Score",
    Scores.DSC_COMP: "Dice Score Complete",
    Scores.DSC_CORE: "Dice Score Core",
    Scores.DSC_EN: "Dice Score Enhancing",
    Scores.DSCP: "Dice Score per Patient",
    Scores.DSCP_COMP: "Dice Score Complete per Patient",
    Scores.DSCP_CORE: "Dice Score Core per Patient",
    Scores.DSCP_EN: "Dice Score Enhancing per Patient",
    Scores.IOUP: "IoU per Patient",
    Scores.LR: "Learning-Rate",
    Scores.L1: "L1-Weight",
    Scores.L2: "L2-Weight",
    Scores.DLTL: "DLTL",
    Scores.DTL : "DTL",
}
