import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import matplotlib

not_succ = []
dir = "/home/christian/Dokumente/Masterthesis/MasterThesis/LateX/figures/graphs"
if not os.path.exists(dir):
    os.makedirs(dir)

fig_height = 5
fig_width = 12

nr_samples = [2, 4, 8, 16, 24, 32]

res_avg_file = "results_new.txt"
res_pp_file = "results_per_patient_new.csv"
marker = ["o", "s", "D", "^", "*"]


def set_big_font():
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 22}
    matplotlib.rc('font', **font)


def set_normal_font():
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 19}
    matplotlib.rc('font', **font)

def set_small_font():
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

def create_and_save_line_plot(x, y, lables, name, size=[12, 8]):
    fig, ax = plt.subplots()
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1])
    xt = np.arange(len(nr_samples))
    ma = min(1.0, np.array(y).max() + 0.02)
    mi = max(0.0, np.array(y).min() - 0.02)
    for i, series in enumerate(y):
        ax.plot(x, series, label=lables[i], marker=marker[i], linewidth=2, markersize=7)

    ax.set_ylabel('DSC')
    ax.set_xlabel('Nr. Training Patients')
    ax.set_xticks(nr_samples)
    ax.set_xticklabels(nr_samples)
    ax.set_ylim([mi, ma])
    ax.legend(loc="lower right")
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "{}.png".format(name)))


def create_and_save_bar_plot(x, y, lables, name, size=[12, 8]):
    fig, ax = plt.subplots()
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1])
    ma = min(1.0, np.array(y).max() + 0.02)
    mi = max(0.0, np.array(y).min() - 0.02)
    xt = np.arange(len(nr_samples))
    width = 0.8 / float(len(y))
    if len(y) == 4:
        rects1 = ax.bar(xt - width * 1.5, dscavg[0], width, label=methods[0])
        rects2 = ax.bar(xt - width / 2, dscavg[1], width, label=methods[1])
        rects3 = ax.bar(xt + width / 2, dscavg[2], width, label=methods[2])
        rects4 = ax.bar(xt + width * 1.5, dscavg[3], width, label=methods[3])
    if len(y) == 5:
        rects1 = ax.bar(xt - width * 2, dscavg[0], width, label=methods[0])
        rects2 = ax.bar(xt - width, dscavg[1], width, label=methods[1])
        rects3 = ax.bar(xt, dscavg[2], width, label=methods[2])
        rects4 = ax.bar(xt + width, dscavg[3], width, label=methods[3])
        rects5 = ax.bar(xt + width * 2, dscavg[4], width, label=methods[4])
    ax.set_ylabel('DSC')
    ax.set_xlabel('Nr. Training Patients')
    ax.set_xticks(xt)
    ax.set_xticklabels(nr_samples)
    ax.set_ylim([mi, ma])
    ax.legend(loc="lower right")
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "{}.png".format(name)))


def create_and_save_scatter_plot(pre, sens, lables, name, size=[10, 10], mean=True, legend=True):
    if not legend:
        size[1] = 7
    fig, ax = plt.subplots()
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1])
    if legend:
        mi = max(0.0, min(np.array(pre).min() - 0.2, np.array(sens).min() - 0.2))
        ma = min(1.0, max(np.array(pre).max() + 0.2, np.array(sens).max() + 0.2))
    else:
        mi = max(0.0, np.array(pre).min() - 0.2)
        ma = min(1.0, np.array(pre).max() + 0.2)
    for i, series in enumerate(sens):
        p = np.mean(pre[i]) if mean else pre[i]
        s = np.mean(series) if mean else series
        ax.scatter(p, s, label=lables[i], marker=marker[i], s=200 if mean else 100)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Sensitivity')
    ax.set_xlabel('Precision')
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    if legend:
        ax.legend()
        ax.set_xlim([mi, ma])
        ax.set_ylim([mi, ma])
    else:
        ax.set_ylim([0.58, 0.93])
        ax.set_xlim([0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "{}.png".format(name)))


def create_and_save_box_plot(score, lables, name, size=[12, 5]):
    set_small_font()
    fig, ax = plt.subplots()
    fig.set_figwidth(size[0])
    fig.set_figheight(size[1])
    ax.set_ylabel('DSC')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.8)
    ax.boxplot(score, showmeans=False, labels=lables)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "{}.png".format(name)))
    set_normal_font()

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
            #print(full_path4)
            fold = int(full_path4.split("_")[-1][1])
            nr_pat = int(full_path4.split("_")[-2])
            result_path = os.path.join(full_path4, "Test_{}/{}".format(fold, res_avg_file))
            result_pp_path = os.path.join(full_path4, "Test_{}/{}".format(fold, res_pp_file))
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

set_normal_font()
create_and_save_bar_plot(nr_samples, dscavg, methods, name="DSC_complete_bar_graph")
create_and_save_line_plot(nr_samples, dscavg, methods, name="DSC_complete_line_graph")
create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_complete", mean=False)
create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_complete_mean")

set_big_font()
create_and_save_bar_plot(nr_samples, dscavg, methods, name="DSC_complete_bar_graph_big")
create_and_save_line_plot(nr_samples, dscavg, methods, name="DSC_complete_line_graph_big")
create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_complete_big", mean=False)
create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_complete_mean_big")

for idx, nr_s in enumerate(nr_samples):
    score = [np.array(dsc_pp[methods[0]][idx]).flatten(), np.array(dsc_pp[methods[1]][idx]).flatten(), np.array(dsc_pp[methods[2]][idx]).flatten(),
             np.array(dsc_pp[methods[3]][idx]).flatten()]
    create_and_save_box_plot(score, methods_multi_line, name="DSC_boxplot_complete_{}".format(nr_s))

score = [np.array(dsc_pp[methods[0]]).mean(axis=0).flatten(),
         np.array(dsc_pp[methods[1]]).mean(axis=0).flatten(),
         np.array(dsc_pp[methods[2]]).mean(axis=0).flatten(),
        np.array(dsc_pp[methods[3]]).mean(axis=0).flatten()]
#score = [np.median(np.array(dsc_pp[methods[0]]),axis=0).flatten(),
#         np.median(np.array(dsc_pp[methods[1]]),axis=0).flatten(),
#         np.median(np.array(dsc_pp[methods[2]]),axis=0).flatten(),
#        np.median(np.array(dsc_pp[methods[3]]), axis=0).flatten()]
create_and_save_box_plot(score, methods_multi_line, name="DSC_boxplot_complete_avg", size=[10, 5])

plt.close('all')

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
        row_i = 1
        d_key = "_COMP"
        yrange = [0.75, 0.9]
    if mask == "CORE":
        row_i = 2
        d_key = "_CORE"
        yrange = [0.6, 0.75]
    if mask == "ENHANCING":
        row_i = 3
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
                #print(full_path4)
                fold = int(full_path4.split("_")[-1][1])
                nr_pat = int(full_path4.split("_")[-2])
                result_path = os.path.join(full_path4, "Test_{}/{}".format(fold, res_avg_file))
                result_pp_path = os.path.join(full_path4, "Test_{}/{}".format(fold, res_pp_file))
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

    set_normal_font()
    create_and_save_bar_plot(nr_samples, dscavg, methods, name="DSC_all_{}_bar_graph".format(mask))
    create_and_save_line_plot(nr_samples, dscavg, methods, name="DSC_all_{}_line_graph".format(mask))
    create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_all_{}".format(mask), mean=False)
    create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_all_{}_mean".format(mask))

    set_big_font()
    create_and_save_bar_plot(nr_samples, dscavg, methods, name="DSC_all_{}_bar_graph_big".format(mask))
    create_and_save_line_plot(nr_samples, dscavg, methods, name="DSC_all_{}_line_graph_big".format(mask))
    create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_all_{}_big".format(mask), mean=False, legend=False)
    create_and_save_scatter_plot(preavg, seavg, methods, name="recall_vs_sens_all_{}_mean_big".format(mask), legend=False)

    for idx, nr_s in enumerate(nr_samples):
        score = [np.array(dsc_pp[methods[0]][idx]).flatten(), np.array(dsc_pp[methods[1]][idx]).flatten(),
                 np.array(dsc_pp[methods[2]][idx]).flatten(),
                 np.array(dsc_pp[methods[3]][idx]).flatten(), np.array(dsc_pp[methods[4]][idx]).flatten()]
        create_and_save_box_plot(score, methods_multi_line, name="DSC_boxplot_all_{}_{}".format(mask, nr_s))

    #score = [np.array(dsc_pp[methods[0]]).mean(axis=0).flatten(),
    #         np.array(dsc_pp[methods[1]]).mean(axis=0).flatten(),
    #         np.array(dsc_pp[methods[2]]).mean(axis=0).flatten(),
    #         np.array(dsc_pp[methods[3]]).mean(axis=0).flatten(),
    #        np.array(dsc_pp[methods[4]]).mean(axis=0).flatten()]
    score = [np.median(np.array(dsc_pp[methods[0]]) ,axis=0).flatten(),
             np.median(np.array(dsc_pp[methods[1]]), axis=0).flatten(),
             np.median(np.array(dsc_pp[methods[2]]), axis=0).flatten(),
             np.median(np.array(dsc_pp[methods[3]]), axis=0).flatten(),
            np.median(np.array(dsc_pp[methods[4]]), axis=0).flatten()]
    create_and_save_box_plot(score, methods_multi_line, name="DSC_boxplot_all_{}_avg".format(mask))

    plt.close('all')

print("Finished with {} errors.".format(len(not_succ)))
print("Errors in {}".format(not_succ))