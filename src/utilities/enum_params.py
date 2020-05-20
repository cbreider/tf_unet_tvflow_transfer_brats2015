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
    CNN Type
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
    COMPLETE = 1    # Complete Tumor mask:  Label 1 or 2 or 3 or 4
    CORE = 2        # Core Tumor mask:      Label 1 or      3 or 4
    ENHANCING = 3   # Enhancing Tumor mask: Label                4
    ALL = 4         # All Classes: Train to predict all 5 classes (0-4) individually


class TrainingModes(Enum):
    """
    training modes
    """
    TVFLOW_REGRESSION = 1
    TVFLOW_SEGMENTATION = 2
    BRATS_SEGMENTATION = 3
    AUTO_ENCODER = 4
    DENOISING_AUTOENCODER = 5
    TVFLOW_SEGMENTATION_TV_PSEUDO_PATIENT = 6


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
    SOFTMAX = 3
    NONE = 4


class RestoreMode(Enum):
    """
    training optimzer
    """
    COMPLETE_SESSION = 1    # complete session including hyper params
    ONLY_GIVEN_VARS = 2       # import only certain varibales from tf checkpoint
    COMPLETE_NET = 3        # continue using the same path to store as restore path


class TV_clustering_method(Enum):
    """
    modes for clustering tv smoothed images
    """
    STATIC_BINNING = 1      # cluster by deving into equal bins
    STATIC_CLUSTERS = 2     # cluster using pre given cluster centers
    K_MEANS = 3             # use k-means to cluster
    MEAN_SHIFT = 4          # use mean shift to cluster


class Scores(Enum):
    """
    Types of Scores measured for CNN performance and other values logged into the summary
    """
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
    VALSCORE = "ValScore"
    PRECISION_CORE = "Prec Core"
    PRECISION_COMP = "Prec Comp."
    PRECISION_EN = "Prec Enhan."
    SENSITIVITY_CORE = "Sens. Core"
    SENSITIVITY_COMP = "Sens. Comp."
    SENSITIVITY_EN = "Sens. Enhan."
    SPECIFICITY_CORE = "Spec Core"
    SPECIFICITY_COMP = "Spec Comp."
    SPECIFICITY_EN = "Spec Enhan."
    PRECISION = "Prec"
    SENSITIVITY = "Sens"
    SPECIFICITY = "Spec"


"""
   Full names for score Enums
"""
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
    Scores.VALSCORE: "Validation Score",
    Scores.PRECISION_CORE: "Precision Core",
    Scores.PRECISION_COMP: "Precision Complete",
    Scores.PRECISION_EN: "Precision Enhancing",
    Scores.SENSITIVITY_CORE: "Sensitivity Core",
    Scores.SENSITIVITY_COMP: "Sensitivity Comp.",
    Scores.SENSITIVITY_EN: "Sensitivity Enhancing",
    Scores.SPECIFICITY_CORE: "Specificity Core",
    Scores.SPECIFICITY_COMP: "Specificity Comp.",
    Scores.SPECIFICITY_EN: "Specificity Enhancing",
    Scores.PRECISION: "Precision",
    Scores.SENSITIVITY: "Sensitivity",
    Scores.SPECIFICITY: "Specificity",
}
