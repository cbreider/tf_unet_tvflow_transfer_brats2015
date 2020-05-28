import os
import subprocess
import  configuration as config
from src.utilities.enum_params import Subtumral_Modes


not_succ = []
for path in os.listdir("./tf_model_output"):
    if "BRATS_segmentation_all_labels" in path or "BRATS_segmentation_complete_mask" in path:
        full_path1 = os.path.join("/home/christian/Data_5/Projects/unet_brats2015/tf_model_output", path)
        for path2 in os.listdir(full_path1):
            full_path2 = os.path.join(full_path1, path2)
            for path3 in os.listdir(full_path2):
                full_path3 = os.path.join(full_path2, path3)
                for path4 in os.listdir(full_path3):
                    full_path4 = os.path.join(full_path3, path4)
                    print(full_path4)
                    fold = full_path4.split("_")[-1][1]
                    if "complete" in full_path4 and "all" in full_path4:
                        not_succ.append(full_path4)
                        continue
                    if "complete" in full_path4:
                        config.Subtumral_Modes = Subtumral_Modes.COMPLETE
                    elif "all" in full_path4:
                        config.Subtumral_Modes = Subtumral_Modes.ALL
                    else:
                        not_succ.append(full_path4)
                        continue
                    subprocess.call(['python3', 'train.py', '--mode', "3", '--cuda_device', "0", '--take_fold_nr', str(fold), '--nr_training_scans', "1", '--restore_mode',
                              "1", '--restore_path', str(full_path4), '--reuse_out_folder', '--include_testing'])
                    if not os.path.exists(os.path.join(full_path4, "Test_{}/results_new.txt".format(fold))):
                        not_succ.append(full_path4)


print(not_succ)