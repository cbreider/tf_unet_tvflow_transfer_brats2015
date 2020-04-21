# Learning Feature Preserving Smoothing as Prior for Image Segmentation (U-Net, Total Variation smoothing)

Master Project & Lab Visualization & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created  2019-2020

Python3 Tool to pre-train a CNN Model on Total Variation smoothed images (regression with MSE), to enhance segmentation performance.
It is also possible to pre-train the CNN in an auto encoder fashion.

Currently only 2D U-Net with Brats 2015 data set are implemented

This Projects is still under development so check out for changes some time.
## Requirements
tool
Written in python 3.5

Necessary packages
  - numpy
  - pypng
  - SimpleITK
  - tensorflow 1.12.0
  - PIL

## Training


The *script train.py* trains Unet on Brats 2015 dataset either with Segmentation ground Truth or tv smoothed data. It should be pre trained with tv flow Data. But can also be trained from scratch. Both scripts require the dataset  in PNG format. MHA format is also possible but not recommended because it could not be loaded with tensorflow datapipeline.

### Arguments

Both scripts offer the following (optional) arguments:
* --mode  [mode] Mode of training session. 1=TV regression training, 2=TV clustering training3=BRATS Segmentation
* --create_new_split (created a new training/Validation/(testing) split. If not given it will create a new on based on the split ratio given in *configuration.py*. It will save Default is False  (optional)
* --do_not_create_summaries: Disable creation of tensorflow summaries. Default False  (optional)
* --restore_path=[PATH]: Restore a tensorflow model/session from a given path. If not given it will create a new session.  (optional)
* --restore_mode [1/2/3]: Mode to restore tensorflow checkpoint. Only used if also restore_path is given. 1 for complete Session. 2 For complete Unet only layers given in *configuration.py*. 3 for complete net (e.g. use other learning rate etc.). Default is 1  (optional)
* --caffemodel_path=[PATH.h5]: Load weights form pre trained caffemodel in hdf5 format. Used for importing pre trained weight from 2d cell segmentation of Uni Freiburg. But it is also possible to import other caffe models of Unet. To do so you may have to configure the layer names. Please have a look at src/tf_unet/caffe2tensorflow_mapping.py  (optional)
* --cuda_device [NR]: Use a specific Cuda device (optional)
* --name [NAME]: Name to use output path. If not given a string containing training mode and timestamp will be used  (optional)
* --take_fold_nr [NR]: Read split from x-fold file with File nr [NR] (*fold[NR].json). If combined with *--create_new_split* 
a new x-fold split is created with x = *nr_of_folds* defined in *configuration.py*  (optional)
* --include_testing: Run Evaluation on the testing split after training.  (optional)
* --data_path [PATH]: Path to the Brats training dataset. Files have to be 2D images and ordered in the same way'
                             '(/2d/slices/png/raw/train/HGG data_path --> Patient --> modality--> *.png). Default ../dataset/  (optional)
* --nr_training_scans: Use only a specific number of all training samples (optional)
*--reuse_out_folder: If *restore_path* is given reuse the folder to save model and continue tf summary

## Testing

The script *predict_brats_seg.py* runs a trained model on the Brats test set or a given test split and saves the 
predicted scans as MHA (for BRATS2015 challenge) and as PNG files (optional). It requires the following arguments:


* --model_path: Path to the (trained) tf model checkpoint
* --name [NAME]: Name for the test run. The output siles will be saved as "VSD.[name].PATIENTID.mha" 
* --cuda_device [NR]: Use a specific cuda device )(optional)
* --save_pngs: save predictions also as PNG images
* --take_fold_nr [NR]: use the test split from a given fold. *fold[NR],json*

## Provide Data
The Brats dataset should be provided as follows:

  - Image Scan data / Segmentation ground truth data: *./../Dataset/2D-Slices/png/raw/...[Default Brats2015 path schema] *
  - You can easily provide your own paths. To do so please have a look at *src/utils/path_utils.py*
  - Split configuration as json: *./../Dataset/splits/...*. The split could be provided as single split "split.json" 
  or k-fold splits ("fold1.json", ..., "fold[k],json). For futher details see *src/utilities/split_utilities.py*
  - All Data pre-processing, augmentation and smoothing is done on the fly with TF Datapipeline on the CPU. Please see 
  *src/tf_data_pipeline_wrapper.py* and *src/tf_data_generator.py*. Currently it is only possible to load the data from 
  PNG images. Therefore the script *convert_brats2015_to_png.py* is provided to convert the BRATS2015 dataset from 3D MHA
  scans to 2D axial PNG slices. Secondly mean, max and vaiance of each scan are stored as *values.json* file in the
  corresponding folder. the
 


 ## Options and Configuration
 
There are a lots of configuration options for the U-Net, Training, data processing and augmentation. Please have a look at *configuration.py* to make sure you have chosen the right configurations.
