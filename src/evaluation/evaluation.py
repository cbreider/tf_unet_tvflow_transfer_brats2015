import numpy as np
import csv
import matplotlib.pyplot as plt
import os

not_succ = []
dir = "./results"
if not os.path.exists(dir):
    os.makedirs(dir)

nr_samples = [2, 4, 8, 16, 24, 32]


"""
#################################################################################################################################################
EVALUATION COMPLETE MASK
#################################################################################################################################################
"""


methods = ["Trained from Scratch",  "Fine-Tuned TV Regression Single Scale", "Fine-Tuned TV Regression Multi Scale",
           "Fine-Tuned Clustered TV Segmentation"]
methods_multi_line = ["Trained from\nScratch",  "Fine-Tuned\nTV Regression\nSingle Scale", "Fine-Tuned\nTV Regression\nMulti Scale",
           "Fine-Tuned\nClustered TV\nSegmentation"]
           
dsc = {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
dsc_pp = {methods[0]: [[], [], [],
                       [], [], []],
       methods[1]:   [[], [], [],
                      [], [], []],
       methods[2]: [[], [], [],
                    [], [], []],
       methods[3]: [[], [], [],
                    [], [], []]}
precision = {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
specificity =  {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
sensitivity =  {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
       methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
full_path1 = "/media/sf_Projects/unet_brats2015/tf_model_output/BRATS_segmentation_complete_mask"
for path in os.listdir(full_path1):
    full_path2 = os.path.join(full_path1, path)
    for path3 in os.listdir(full_path2):
        full_path3 = os.path.join(full_path2, path3)
        for path4 in os.listdir(full_path3):
            full_path4 = os.path.join(full_path3, path4)
            print(full_path4)
            fold = int(full_path4.split("_")[-1][1])
            nr_pat = int(full_path4.split("_")[-2])
            result_path = os.path.join(full_path4, "Test_{}/results.txt".format(fold))
            result_pp_path = os.path.join(full_path4, "Test_{}/results_per_patient.csv".format(fold))
            if not os.path.exists(result_path) or not os.path.exists(result_pp_path):
                print("Results not found for {}!".format(full_path4))
                not_succ.append(full_path4)
                continue
            if "trained_from_scratch" in full_path4:
                type = methods[0]
            elif "tv_regression_single_scale" in full_path4:
                type = methods[1]
            elif "tv_regression_multi_scale" in full_path4:
                type = methods[2]
            elif "tv_clustering_segmentation_single_scale" in full_path4:
                type = methods[3]

            else:
                print("No matching baseline found for {}!".format(full_path4))
                not_succ.append(full_path4)
                continue
            with open(result_pp_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                line_count = 0
                for row in csv_reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        line_count += 1
                        if len(row) < 5:
                            continue
                        dsc_pp[type][nr_samples.index(nr_pat)].append(float(row[0]))
            with open(result_path) as res_file:
                line_count = 0
                for row in res_file:
                    if line_count == 0:
                        line_count += 1
                    else:
                        line_count += 1
                        if "DSCP" in row:
                            dsc[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]
                        if "PRECISION" in row:
                            precision[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]
                        if "SPECIFICITY" in row:
                            specificity[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]
                        if "SENSITIVITY" in row:
                            sensitivity[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]


dscavg = [[], [], [], []]
preavg = [[], [], [], []]
seavg = [[], [], [], []]
spavg = [[], [], [], []]
headline = "Nr. Training Patients;Method;Fold1;;;;Fold2;;;;Fold3;;;;Fold4;;;;Fold5;;;;Average;;;\n"
headline = "{};;Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity" \
         ";Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity\n".format(headline)
csv_out = headline
for idx, nr_sam in enumerate(nr_samples):
    mthc = 0
    for method in methods:
        if mthc == 0:
            row = "{};".format(str(nr_sam))
        else:
            row = ";"
        row = "{}{};".format(row, method)
        d = dsc[method][idx]
        pr = precision[method][idx]
        se = sensitivity[method][idx]
        sp = specificity[method][idx]
        da = np.mean(d)
        pa = np.mean(pr)
        sa = np.mean(se)
        sca = np.mean(sp)
        for f in range(0,5):
            row = "{}{};{};{};{};".format(row, d[f], pr[f], se[f], sp[f])
        row = "{}{};{};{};{}\n".format(row, da, pa, sa, sca)
        dscavg[methods.index(method)].append(da)
        preavg[methods.index(method)].append(pa)
        seavg[methods.index(method)].append(sa)
        spavg[methods.index(method)].append(sca)
        csv_out = "{}{}".format(csv_out, row)

outF = open(dir + "/results_BRATS_complete_tumor_mask.csv", "w")
outF.write(csv_out)
outF.write("\n")
outF.close()

x = np.arange(len(nr_samples))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(9)
rects1 = ax.bar(x - width * 1.5, dscavg[0], width, label=methods[0])
rects2 = ax.bar(x - width / 2, dscavg[1], width, label=methods[1])
rects3 = ax.bar(x + width / 2, dscavg[2], width, label=methods[2])
rects4 = ax.bar(x + width * 1.5, dscavg[3], width, label=methods[3])
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DSC')
ax.set_xlabel('Nr. Training Patients')
ax.set_xticks(x)
ax.set_xticklabels(nr_samples)
ax.set_ylim([0.75, 0.9])
ax.legend(loc="lower right")
plt.savefig(dir + "/DSC_complete_bar_graph.png")


fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(9)
ax.plot(nr_samples, dscavg[0], label=methods[0], marker="o", linewidth=3, markersize=10)
ax.plot(nr_samples, dscavg[1], label=methods[1], marker="s", linewidth=3, markersize=10)
ax.plot(nr_samples, dscavg[2], label=methods[2], marker="D", linewidth=3, markersize=10)
ax.plot(nr_samples, dscavg[3], label=methods[3], marker="^", linewidth=3, markersize=10)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DSC')
ax.set_xlabel('Nr. Training Patients')
ax.set_xticklabels(nr_samples)
ax.set_ylim([0.78, 0.893])
ax.legend(loc="lower right")
plt.savefig(dir + "/DSC_complete_line_graph.png")


fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(9)
aaa = np.mean(seavg[3])
ax.scatter(np.mean(preavg[0]), np.mean(seavg[0]), label=methods[0], marker="o", s=2)
ax.scatter(np.mean(preavg[1]), np.mean(seavg[1]), label=methods[1], marker="s", s=100)
ax.scatter(np.mean(preavg[2]), np.mean(seavg[2]), label=methods[2], marker="D", s=100)
ax.scatter(np.mean(preavg[3]), np.mean(seavg[3]), label=methods[3], marker="^", s=100)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('DSC')
ax.set_xlabel('Nr. Training Patients')
ax.legend(loc="lower right")
#ax.set_ylim([0.7, 0.9])
plt.savefig(dir + "/recall_vs_sens_complete.png")

for idx, nr_s in enumerate(nr_samples):
    score = [np.array(dsc_pp[methods[0]][idx]).flatten(), np.array(dsc_pp[methods[1]][idx]).flatten(), np.array(dsc_pp[methods[2]][idx]).flatten(),
             np.array(dsc_pp[methods[3]][idx]).flatten()]
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(9)
    ax.set_ylabel('DSC')
    ax.boxplot(score, showmeans=True, labels=methods_multi_line)
    plt.savefig(dir + "/DSC_boxplot_complete_{}.png".format(nr_s))

score = [np.array(dsc_pp[methods[0]]).flatten(), np.array(dsc_pp[methods[1]]).flatten(), np.array(dsc_pp[methods[2]]).flatten(),
        np.array(dsc_pp[methods[3]]).flatten()]
fig, ax = plt.subplots()
fig.set_figheight(4)
fig.set_figwidth(9)
ax.set_ylabel('DSC')
ax.boxplot(score, showmeans=True, labels=methods_multi_line)
plt.savefig(dir + "/DSC_boxplot_complete_avg.png")
print(not_succ)


""""
#################################################################################################################################################
EVALUATION ALL LABELS
#################################################################################################################################################
"""

methods = ["Trained from Scratch", "Trained from Scratch TV Pseudo-Patient", "Fine-Tuned Auto-Encoder",
           "Fine-Tuned Denoising Auto-Encoder", "Fine-Tuned TV Regression Multi Scale"]

methods_multi_line = ["Trained from\nScratch", "Trained from Scratch TV\nPseudo-Patient", "Fine-Tuned\nAuto-Encoder",
                      "Fine-Tuned\nDenoising Auto-Encoder", "Fine-Tuned\nTV Regression\nMulti Scale"]


for mask in ["COMPLETE", "CORE", "ENHANCING"]:
    if mask == "COMPLETE":
        row_i = 0
        d_key = "_COMP"
        yrange = [0.75, 0.9]
    if mask == "CORE":
        row_i = 1
        d_key = "_CORE"
        yrange = [0.6, 0.75]
    if mask == "ENHANCING":
        row_i = 2
        d_key = "_EN"
        yrange = [0.6, 0.75]
    dsc = {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
            methods[4]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}

    dsc_pp = {methods[0]: [[], [], [],
                           [], [], []],
           methods[1]:   [[], [], [],
                          [], [], []],
           methods[2]: [[], [], [],
                        [], [], []],
           methods[3]: [[], [], [],
                        [], [], []],
           methods[4]: [[], [], [],
                        [], [], []]}

    precision = {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
            methods[4]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
    specificity =  {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
            methods[4]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
    sensitivity =  {methods[0]: np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[1]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[2]:   np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
           methods[3]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]]),
            methods[4]:  np.array([[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])}
    full_path1 = "/media/sf_Projects/unet_brats2015/tf_model_output/BRATS_segmentation_all_labels"
    for path in os.listdir(full_path1):
        full_path2 = os.path.join(full_path1, path)
        for path3 in os.listdir(full_path2):
            full_path3 = os.path.join(full_path2, path3)
            for path4 in os.listdir(full_path3):
                full_path4 = os.path.join(full_path3, path4)
                print(full_path4)
                fold = int(full_path4.split("_")[-1][1])
                nr_pat = int(full_path4.split("_")[-2])
                result_path = os.path.join(full_path4, "Test_{}/results.txt".format(fold))
                result_pp_path = os.path.join(full_path4, "Test_{}/results_per_patient.csv".format(fold))
                if not os.path.exists(result_path) or not os.path.exists(result_pp_path):
                    print("Results not found for {}!".format(full_path4))
                    not_succ.append(full_path4)
                    continue
                if "trained_from_scratch_base" in full_path4:
                    type = methods[0]
                elif "trained_from_scratch_tv_pseudo_patient" in full_path4:
                    type = methods[1]
                elif "/auto_encoder/" in full_path4:
                    type = methods[2]
                elif "denoising_auto_encoder" in full_path4:
                    type = methods[3]
                elif "tv_regression_multi_scale" in full_path4:
                    type = methods[4]
                else:
                    print("No matching baseline found for {}!".format(full_path4))
                    not_succ.append(full_path4)
                    continue
                with open(result_pp_path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=';')
                    line_count = 0
                    for row in csv_reader:
                        if line_count == 0:
                            line_count += 1
                        else:
                            line_count += 1
                            if len(row) < 3:
                                continue
                            dsc_pp[type][nr_samples.index(nr_pat)].append(float(row[row_i]))
                with open(result_path) as res_file:
                    line_count = 0
                    for row in res_file:
                        if line_count == 0:
                            line_count += 1
                        else:
                            line_count += 1
                            if "DSCP"+d_key in row:
                                dsc[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]
                            if "PRECISION"+d_key in row:
                                precision[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]
                            if "SPECIFICITY"+d_key in row:
                                specificity[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]
                            if "SENSITIVITY"+d_key in row:
                                sensitivity[type][nr_samples.index(nr_pat), fold-1] = row.split(" ")[-1]


    dscavg = [[], [], [], [], []]
    preavg = [[], [], [], [], []]
    seavg = [[], [], [], [], []]
    spavg = [[], [], [], [], []]
    headline = "Nr. Training Patients;Method;Fold1;;;;Fold2;;;;Fold3;;;;Fold4;;;;Fold5;;;;Average;;;\n"
    headline = "{};;Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity" \
             ";Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity;Dice;Precision;Sensitivity;Specificity\n".format(headline)
    csv_out = headline
    for idx, nr_sam in enumerate(nr_samples):
        mthc = 0
        for method in methods:
            if mthc == 0:
                row = "{};".format(str(nr_sam))
            else:
                row = ";"
            row = "{}{};".format(row, method)
            d = dsc[method][idx]
            pr = precision[method][idx]
            se = sensitivity[method][idx]
            sp = specificity[method][idx]
            da = np.mean(d)
            pa = np.mean(pr)
            sa = np.mean(se)
            sca = np.mean(sp)
            for f in range(0,5):
                row = "{}{};{};{};{};".format(row, d[f], pr[f], se[f], sp[f])
            row = "{}{};{};{};{}\n".format(row, da, pa, sa, sca)
            dscavg[methods.index(method)].append(da)
            preavg[methods.index(method)].append(pa)
            seavg[methods.index(method)].append(sa)
            spavg[methods.index(method)].append(sca)
            csv_out = "{}{}".format(csv_out, row)

    outF = open(dir + "/results_BRATS_all_{}_labels.csv".format(mask), "w")
    outF.write(csv_out)
    outF.write("\n")
    outF.close()

    x = np.arange(len(nr_samples))  # the label locations
    width = 0.16  # the width of the bars

    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(9)
    rects1 = ax.bar(x - width * 2, dscavg[0], width, label=methods[0])
    rects2 = ax.bar(x - width, dscavg[1], width, label=methods[1])
    rects3 = ax.bar(x, dscavg[2], width, label=methods[2])
    rects4 = ax.bar(x + width, dscavg[3], width, label=methods[3])
    rects5 = ax.bar(x + width * 2, dscavg[4], width, label=methods[4])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('DSC')
    ax.set_xlabel('Nr. Training Patients')
    ax.set_xticks(x)
    ax.set_xticklabels(nr_samples)
    ax.set_ylim(yrange)
    ax.legend(loc="lower right")
    plt.savefig(dir + "/DSC_all_{}_bar_graph.png".format(mask))


    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(9)
    ax.plot(nr_samples, dscavg[0], label=methods[0], marker="o", linewidth=3, markersize=10)
    ax.plot(nr_samples, dscavg[1], label=methods[1], marker="s", linewidth=3, markersize=10)
    ax.plot(nr_samples, dscavg[2], label=methods[2], marker="D", linewidth=3, markersize=10)
    ax.plot(nr_samples, dscavg[3], label=methods[3], marker="^", linewidth=3, markersize=10)
    ax.plot(nr_samples, dscavg[4], label=methods[4], marker="^", linewidth=3, markersize=10)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('DSC')
    ax.set_xlabel('Nr. Training Patients')
    ax.set_xticklabels(nr_samples)
    ax.set_ylim(yrange)
    ax.legend(loc="lower right")
    plt.savefig(dir + "/DSC_all_{}_line_graph.png".format(mask))


    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(9)
    aaa = np.mean(seavg[3])
    ax.scatter(np.mean(preavg[0]), np.mean(seavg[0]), label=methods[0], marker="o", s=100)
    ax.scatter(np.mean(preavg[1]), np.mean(seavg[1]), label=methods[1], marker="s", s=100)
    ax.scatter(np.mean(preavg[2]), np.mean(seavg[2]), label=methods[2], marker="D", s=100)
    ax.scatter(np.mean(preavg[3]), np.mean(seavg[3]), label=methods[3], marker="^", s=100)
    ax.scatter(np.mean(preavg[4]), np.mean(seavg[4]), label=methods[4], marker="^", s=100)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('DSC')
    ax.set_xlabel('Nr. Training Patients')
    ax.legend(loc="lower right")
    #ax.set_ylim([0.7, 0.9])
    plt.savefig(dir + "/recall_vs_sens_all_{}.png".format(mask))

    for idx, nr_s in enumerate(nr_samples):
        score = [np.array(dsc_pp[methods[0]][idx]).flatten(), np.array(dsc_pp[methods[1]][idx]).flatten(), np.array(dsc_pp[methods[2]][idx]).flatten(),
                 np.array(dsc_pp[methods[3]][idx]).flatten(), np.array(dsc_pp[methods[4]][idx]).flatten()]
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(9)
        ax.set_ylabel('DSC')
        ax.boxplot(score, showmeans=True, labels=methods_multi_line)
        plt.savefig(dir + "/DSC_boxplot_all_{}_{}.png".format(mask, nr_s))

    score = [np.array(dsc_pp[methods[0]]).flatten(), np.array(dsc_pp[methods[1]]).flatten(), np.array(dsc_pp[methods[2]]).flatten(),
            np.array(dsc_pp[methods[3]]).flatten(), np.array(dsc_pp[methods[4]]).flatten()]
    fig, ax = plt.subplots()
    fig.set_figheight(7)
    fig.set_figwidth(9)
    ax.set_ylabel('DSC')
    ax.boxplot(score, showmeans=True, labels=methods_multi_line)
    plt.savefig(dir + "/DSC_boxplot_all_{}_avg.png".format(mask))
    plt.close('all')


print(not_succ)
