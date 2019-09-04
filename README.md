# Pre train Unet on Total Variation smoothed Data

Lab Visualization & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created on June 2019


Python3 tool to pre train Unet on total variation smoothed images (regression with MSE), to enhance segmentation performance on brats 2015 dataset.

## Requirements

Written in python 3.5

Necessary packages
  - numpy
  - pypng
  - pynrrd
  - SimpleITK
  - tensorflow 1.12.0
  - sipy
  - PIL

## Training

### Pre Training on TV smoothed data

The Script *tvflow_pre_training.py* can be used to train Unet to reproduce TV smoothed images from original mha data slices. TV Data should be provided as ground truth.

### Segmentation Training

The *script brats_seg_training.py* trains Unet on Brats 2015 dataset. It should be pre trained with tv flow Data. But can also be trained from scratch. Both scripts require the dataset  in PNG format. MHA format is also possible but not recommended because it could not be loaded with tensorflow datapipeline.

### Arguments

Both scripts offer the following (optional) arguments:
* --create_new_split (created a new training/Validation/(testing) split. If not given it will create a new on based on the split ratio given in *configuration.py*. It will save Default is False
* --do_not_create_summaries: Disable creation of tensorflow summaries. Default False
* --restore_path [PATH]: Restore a tensorflow model/session from a given path. If not given it will create a new session.
* --restore_mode [1/2/3]: Mode to restore tensorflow checkpoint. Only used if also restore_path is given. 1 for complete Session. 2 For complete Unet but not output layer(go from regression to classification/segmentation, e.g. switching from TV pre training to brats segmentation training) 3 for complete net (e.g. use other learning rate etc.). Default is 1
* --caffemodel_path [PATH.h5]: Load weights form pre trained caffemodel in hdf5 format. Used for importing pre trained weight from 2d cell segmentation of Uni Freiburg. But it is also possible to import other caffe models of Unet. To do so you may have to configure the layer names. Please have a look at src/tf_unet/caffe2tensorflow_mapping.py
* --cuda_device [NR]: Use a specific Cuda device



## Testing

The script *predict_brats_seg.py* evaluates the test segmentation performance on a given test split. It requires the following arguments:

* --model_path: Path to the (trained) tf model
* -- cuda_device: Use a specific cuda device )(optional)
' --save_all_predictions: Safe all predictions as PNG. If not given it will only (randomly) save 1/100 of the data.

## Provide Data
The Brats dataset should be provided as follows:

  - Image Scan ata / Segmentation ground truth data: *./../Dataset/2D-Slices/png/raw/...[Default Brats2015 path schema] *
  - TV smoothed Data: *./../Dataset/2D-Slices/png/tvflow/... [Default Brats2015 path schema] *
  - Split configuration as text: *./../Dataset/splits/...*
 
 You can easily provide your own paths. To do so please have a look at *src/utils/path_utils.py*


 ## Options and Configuration
 
There are a lots of configuration options for the Unet, Training, and Data processing. Plea have a look at *configuration.py* to make sure you have chosen the right configurations.
