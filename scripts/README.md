# Scripts to perform experiments for the thesis

To run the experiments please:

 1. Download BraTS2015 training and testing set and unpack them relative to the project root folder as follows 
    1.BraTS training: ./../dataset/BRATS2015_Training/
    2 Brats testing: ./../dataset/BRATS2015_Testing/HGG_LGG
    
 2. Convert Brats data set from MHA to 2D axial PNG slinces by running the *script convert_brats2015_to_png.py* by from the project root folder.
 
 3. Pre-Train prior models (TV and autodencoder) by running the script *pre_train_base_lines.sh*
 
 4. Train models from scratch and models from scratch with Tv pseudo patient with the script *run_five_fold_cross_validation_scratch.sh* in five-fold cross-validation.
 
 5. Fine-tune pre-train models from 3.:
    1. Open the script *run_five_fold_cross_validation_fine_tune.sh* and replace the restore_path with the path of 
    a pre-trained model from yours and replace the name argument with a string of your choice (name for output folder).
    2. Run the script. It will automatically run a validation on the test split after training for each training procedure.
    3. Repeat i. and ii. for all pre-trained models
    
 6. If you want you can use a specific model to precedict on the BRATS challenge set with *predict_brats_seg.py*. 
 The results are stored as MHA files, which can be uploaded to the BRATS competition. 
 The README in the root directory gives further details for this script.
 
 