import numpy as np
import csv
import  matplotlib.pyplot as plt

comp_tv = []
core_tv = []
enh_tv = []
comp_scr = []
core_scr = []
enh_scr = []

for i in range(1, 6):
    filen = "/media/sf_Projects/unet_brats2015/tf_model_output/s_4_f{i}/Test_{i}/results_per_patient.csv".format(i=i)
    with open(filen) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                print(line_count)
                if len(row) < 3:
                    continue
                comp_scr.append(float(row[0]))
                core_scr.append(float(row[1]))
                enh_scr.append(float(row[2]))

for i in range(1, 6):
    filen = "/media/sf_Projects/unet_brats2015/tf_model_output/seg_all_rm_4_f{i}/Test_{i}/results_per_patient.csv".format(i=i)
    with open(filen) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                print(line_count)
                if len(row) < 3:
                    continue
                comp_tv.append(float(row[0]))
                core_tv.append(float(row[1]))
                enh_tv.append(float(row[2]))

comp_scr_np = np.array(comp_scr)
core_scr_np = np.array(core_scr)
enh_scr_np = np.array(enh_scr)
comp_tv_np = np.array(comp_tv)
core_tv_np = np.array(core_tv)
enh_tv_np = np.array(enh_tv)

scores_comp = [comp_scr_np, comp_tv_np]
scores_core = [core_scr_np, core_tv_np]
scores_enh = [enh_scr_np, enh_tv_np]
lables = ["Trained Scratch", "TV pre-trained"]

fig1, ax1 = plt.subplots()
ax1.set_title('Dice Score Complete Mask')
ax1.boxplot(scores_comp, showmeans=True, labels=lables)

fig2, ax2 = plt.subplots()
ax2.set_title('Dice Score Core Mask')
ax2.boxplot(scores_core, showmeans=True, labels=lables)

fig3, ax3 = plt.subplots()
ax3.set_title('Dice Score Enhancing Mask')
ax3.boxplot(scores_enh, showmeans=True, labels=lables)

plt.show()
a = 1
