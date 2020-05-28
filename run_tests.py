import os
from .train import main


for path in os.listdir("./"):
    if "BRATS_segmentation_all_labels" in path or "BRATS_segmentation_complete_mask" in path:
        full_path1 = os.path.join("/home/christian/Data_5/Projects/unet_brats2015/tf_model_output", path)
        for path2 in os.listdir(full_path1):
            full_path2 = os.path.join(full_path1, path2)
            for path3 in os.listdir(full_path2):
                full_path3 = os.path.join(full_path2, path3)
                for path4 in os.listdir(full_path3):
                    full_path4 = os.path.join(full_path3, path4)
                    if "seg_complete_rss_4_f2" in full_path4:
                        print(full_path4)
                        fold = full_path4.split("_")[-1][1]
                        main(['--mode', 3, 'cuda_device', 0, '--fold_nr', fold, 'nr_training_scans', 1, '--restore_mode',
                              1, 'restore_path', full_path4, '--reuse_out_folder', '--include_testing'])