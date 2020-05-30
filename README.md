# Learning Feature Preserving Smoothing as Prior for Image Segmentation (U-Net, Total Variation smoothing)

Master Project & Lab Visualization & Medical Image Analysis SS2019
Institute of Computer Science II

Author: Christian Breiderhoff
created  2019-2020

Python3 Tool to pre-train a CNN Model on Total Variation smoothed images (regression with MSE), to enhance segmentation performance.
It is also possible to pre-train the CNN in an autoencoder fashion.

Currently, only 2D U-Net with BraTS2015 data set are implemented. This project uses the tf.Data API to perform all data preprocessing including TV smoothing on the fly (CPU)

## Requirements
tool
Written in python 3.5

Necessary packages
  * TensorFlow          V1.12.0: Implementation Deep Learning and Data pipeline. TensorFlow 1.12 has to be used because of some incompatibilities of the tf.Data class implementation of our code to newer Versions of TensorFlow
  * Numpy               V1.16.4: Evaluation and array calculation
  * Pillow              V6.0.0: Python Image Library (PIL) fork. Loading and storing images.
  * elasticdeform.tf    V0.4.6: For elastic deformation within the TensorFlow Data-Pipeline [https://github.com/gvtulder/elasticdeform/tree/master/docs/source]
  * h5py                V2.9: For importing variables from Caffe models
  * SimpleITK           V1.2.4: For loading an storing .mha-Files
  * matplotlib          V3.0.3: For some test functions
  * sklearn             V0.23: For some test functions

## Training


The *script train.py* trains U-Net on Brats 2015 dataset either with Segmentation ground Truth or tv smoothed data. It should be pre-trained with tv flow Data. But can also be trained from scratch. Both scripts require the dataset in PNG format. MHA format is also possible but not recommended because it could not be loaded with the TensorFlow data pipeline.

### Arguments

Both scripts offer the following (optional) arguments:
* --mode  [mode] Mode of training session. 1=TV regression training, 2=TV clustering training, 3=brain tumor segmentation on BraTS, 4=Autoencoder (regression training), 5= Denoising autoencoder (regression training), 6=BRATS segmentation with additioanl TV smoothed pseudo patient as data augmentation
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
* --data_path [PATH]: Path to the Brats training dataset. Files have to be 2D images and ordered in the same way *([path]/2d/slices/png/raw/train/HGG data_path --> Patient --> modality-> "sclice".png). Default ../dataset/ * (optional)
* --nr_training_scans: Use only a specific number of all training samples (optional)
*--reuse_out_folder: If *restore_path* is given reuse the folder to save model and continue tf summary

## Prediction and Testing 
If you provide the *--include_testing*  argument for the *train.py* script, it will automatically perform an evaluation on the test split if it is given
The script *predict_brats_seg.py* runs a trained model on the Brats test set or a given test split and saves the predicted scans as MHA (for BRATS2015 challenge) and as PNG files (optional). It requires the following arguments:


* --model_path: Path to the (trained) tf model checkpoint
* --name [NAME]: Name for the test run. The output slices will be saved as "VSD.[name].PATIENTID.mha" 
* --cuda_device [NR]: Use a specific Cuda device )(optional)
* --save_pngs: save predictions also as PNG images
* --take_fold_nr [NR]: use the test split from a given fold. *fold[NR].json*
* --use_brats_test_set: Do inference on the BRATS test set instead of test split of the training set.

## Provide Data
The Brats dataset should be provided as follows:

  * Image Scan data as PNG slices and segmentation ground truth data: *./../dataset/2D_slices/png/raw/...[Default Brats2015 path schema] *
  * You can easily provide your own paths. To do so, please have a look at *src/utils/path_utils.py*
  * Split configuration as json: *./../dataset/splits/...*. The split could be provided as single split "split.json" 
  or k-fold splits ("fold1.json", ..., "fold[k].json). For further details see *src/utilities/split_utilities.py*
  * All Data preprocessing, augmentation and smoothing are done on the fly with TF Datapipeline on the CPU. Please see 
  *src/tf_data_pipeline_wrapper.py* and *src/tf_data_generator.py*. Currently, it is only possible to load the data from 
  PNG images. Therefore the script *convert_brats2015_to_png.py* is provided to convert the BRATS2015 dataset from 3D MHA
  scans to 2D axial PNG slices. Secondly mean, max, and variance of each scan are stored as *values.json* file in the corresponding folder. the
 

The data for the data pipeline has to be provided as PNG images (2D) slices. The script *convert_brats2015_to_png.py* will convert the BRATS data set from 3D MHA files to 2D (axial) PNG slices within the shown data structure above.  It also creates an additional *values.json* file, in which the maximum, mean, and variance of each scan is stored. To successfully run the script, please provide the BraTS data set as follows:

BraTs2015 test set: *./../dataset/BRATS2015_Testing/[Unpacked BraTS test data set starting with "HGG_LGG" folder
BraTs2015 training set: *./../dataset/BRATS2015_Training/[Unpacked BraTS training data set starting with "HGG" and "LGG" folder


 ## Options and Configuration
 
There are a lot of configuration options for the U-Net, Training, data processing, and augmentation. Please have a look at *configuration.py* to make sure you have chosen the right configurations.

## Scripts
The folder *./scripts* contains bash scripts that automatically perform TV/(denoising) autoencoder pre-training, training from scratch and fine-tuning pre-trained models for brain tumor segmentation of BraTS2015. The folder contains an additional Readme file for further information.