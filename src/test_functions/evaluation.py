import numpy as np
import csv
import matplotlib.pyplot as plt

comp = {"s": [], "seg_all_rm": [], "seg_all_aens": [], "seg_all_dae": []}
core = {"s": [], "seg_all_rm": [], "seg_all_aens": [], "seg_all_dae": []}
enh = {"s": [], "seg_all_rm": [], "seg_all_aens": [], "seg_all_dae": []}
for baseline in ["s", "seg_all_rm", "seg_all_aens", "seg_all_dae"]:
    for nr_samples in [2, 4, 8, 16, 24, 32]:
        for i in range(1, 6):
            if baseline == "s" and nr_samples == 32 and i > 2:
                continue
            filen = "/media/sf_Projects/unet_brats2015/tf_model_output/{bs}_{nrs}_f{i}/Test_{i}/results_per_patient.csv".format(i=i,
                                                                                                                                bs=baseline,
                                                                                                                                nrs=nr_samples)
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
                            print("ERROR")
                            continue
                        comp[baseline].append(float(row[0]))
                        core[baseline].append(float(row[1]))
                        enh[baseline].append(float(row[2]))



scores_comp = [np.array(comp["s"]), np.array(comp["seg_all_aens"]), np.array(comp["seg_all_dae"]), np.array(comp["seg_all_rm"])]
scores_core = [np.array(core["s"]), np.array(core["seg_all_aens"]), np.array(core["seg_all_dae"]), np.array(core["seg_all_rm"])]
scores_enh = [np.array(enh["s"]), np.array(enh["seg_all_aens"]), np.array(enh["seg_all_dae"]), np.array(enh["seg_all_rm"])]
lables = ["Scratch", "AE pre-trained", "Den. AE pre-trained", "TV pre-trained"]

fig1, ax1 = plt.subplots()
ax1.set_title('Dice Score Complete Mask')
ax1.boxplot(scores_comp, sym='', showmeans=True, labels=lables)
plt.savefig('DSC_complete_32.png')

fig2, ax2 = plt.subplots()
ax2.set_title('Dice Score Core Mask')
ax2.boxplot(scores_core, sym='', showmeans=True, labels=lables)
plt.savefig('DSC_core_32.png')
fig3, ax3 = plt.subplots()
ax3.set_title('Dice Score Enhancing Mask')
ax3.boxplot(scores_enh, sym='', showmeans=True, labels=lables)
plt.savefig('DSC_enhancing_32.png')
plt.show()
a = 1
