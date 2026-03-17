import matplotlib.pyplot as plt
import numpy as np
import os
import json
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from src.config import OUT_PATH, col_pc
from src.setting import ExcludSubj
from src.decomposition import prep_data_trial
from src.analysis import compute_tr_gc

tfr_path = OUT_PATH+ '/Data_longWOBS'
subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
subj_included = ExcludSubj(subj_included, data_path=tfr_path)

path = tfr_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time_tfr=d['time_tfr']

    
band = 'high_gamma'
method_pca = 'concat'
data_aug_method = 'duplicat'

window_len, step_len, maxlag = 60, 6, None

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[2/5, 3/5])
ax_TS = fig.add_subplot(gs[0, :])  
ax_TS.set_ylabel('PCs')
ax_TS.set_xlabel('Time (s)')
ax_TS.grid()

ax = fig.add_subplot(gs[1, 0])
ax_old = fig.add_subplot(gs[1, 1])
ax_new = fig.add_subplot(gs[1, 2])

axes = [ax, ax_old, ax_new]
name_lab = ['full', 'old', 'new']
marker = ['x', 'o', '^']

lab = ['PC1', 'PC3']

x2y_sh_m= {0:[], 1:[], 2:[]}
y2x_sh_m={0:[], 1:[], 2:[]}
x2y_sh_std={0:[], 1:[], 2:[]}
y2x_sh_std={0:[], 1:[], 2:[]}
x2y_sh_per={0:[], 1:[], 2:[]}
y2x_sh_per={0:[], 1:[], 2:[]}

gc_x2y_all= {0:[], 1:[], 2:[]}
gc_x2y_max = {0:[], 1:[], 2:[]}
gc_y2x_all= {0:[], 1:[], 2:[]}
gc_y2x_max= {0:[], 1:[], 2:[]}

lags = [3, 4, 5, 6, 7, 8, 9, 10] # 15ms, 20ms, 25ms, ... 50ms

for s in range(30) : 
    X_0,  X_0_old, X_0_new, y_0 = prep_data_trial(band, method_pca, data_aug_method, subj_included, 0,tfr_path)
    X_1,  X_1_old, X_1_new, y_1 = prep_data_trial(band, method_pca, data_aug_method, subj_included, 1,tfr_path)
    X_2,  X_2_old, X_2_new, y_2 = prep_data_trial(band, method_pca, data_aug_method, subj_included, 2,tfr_path)

    if s == 0 : 
        ax_TS.plot(time_tfr, np.mean(X_0, axis=0), ls='-', c=col_pc[0], label=lab[0])
        ax_TS.plot(time_tfr, np.mean(X_2, axis=0), ls='-', c=col_pc[2], label=lab[1])

    for i, (x1, x2) in enumerate([(X_0, X_2), (X_0_old, X_2_old), (X_0_new, X_2_new)]):
        if maxlag == None : 
            if s==0 and i == 0 : 
                # optimize the lag on full :
                bics = []
                for l in lags :
                    _, _, b = compute_tr_gc(x=x1, y=x2, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=l)
                    bics.append(np.mean(b))

                maxlag = lags[np.argmin(bics)]

        gc_x2y, times, _ = compute_tr_gc(x=x1, y=x2, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
        gc_y2x, _, _ = compute_tr_gc(x2, x1, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
        gc_x2y_all[i].append(gc_x2y)
        gc_y2x_all[i].append(gc_y2x)
        
        gc_x2y_sh = []
        gc_y2x_sh = []
        for perm in range(100) :
            x1_sh = np.random.permutation(x1.T).T
            x2_sh = np.random.permutation(x2.T).T
            x2y, _, _ = compute_tr_gc(x1_sh, x2_sh, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
            y2x, _, _ = compute_tr_gc(x2_sh, x1_sh, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
            gc_x2y_sh.append(x2y)
            gc_y2x_sh.append(y2x)
        x2y_sh = np.vstack(gc_x2y_sh)
        y2x_sh = np.vstack(gc_y2x_sh)

        x2y_sh_m[i].append(np.mean(x2y_sh, axis=0))
        y2x_sh_m[i].append(np.mean(y2x_sh, axis=0))
        x2y_sh_std[i].append(np.std(x2y_sh, axis=0))
        y2x_sh_std[i].append(np.std(y2x_sh, axis=0))
        x2y_sh_per[i].append(np.percentile(x2y_sh,95, axis=0))
        y2x_sh_per[i].append(np.percentile(y2x_sh,95, axis=0))

# plot
for i in range(3):
    gc_x2y=np.mean(np.vstack(gc_x2y_all[i]), axis=0)
    gc_y2x=np.mean(np.vstack(gc_y2x_all[i]), axis=0)

    axes[i].plot(times, gc_x2y, label=f'{lab[0]} → {lab[1]}', c='grey', ls='-')
    axes[i].plot(times, gc_y2x, label=f'{lab[1]} → {lab[0]}',  c='k', ls='-')
    axes[i].plot(times, np.mean(x2y_sh_m[i], axis=0), label=f'{lab[0]} → {lab[1]} shuffled', c='grey', ls='--', linewidth = 0.5)
    axes[i].plot(times, np.mean(y2x_sh_m[i], axis=0), label=f'{lab[1]} → {lab[0]} shuffled',  c='k', ls='--', linewidth = 0.5)
    axes[i].fill_between(times, np.mean(x2y_sh_m[i], axis=0) + np.mean(x2y_sh_m[i], axis=0), np.mean(x2y_sh[i],  axis=0) - np.mean(x2y_sh_m[i],  axis=0), color='grey', alpha = 0.1)
    axes[i].fill_between(times, np.mean(y2x_sh_m[i], axis=0) + np.mean(y2x_sh_std[i],  axis=0), np.mean(y2x_sh_m[i],  axis=0) - np.mean(y2x_sh_std[i],  axis=0),  color='k',  alpha = 0.1)
    axes[i].plot(times, np.mean(x2y_sh_per[i],  axis=0), c='grey', ls=':', linewidth = 0.8)
    axes[i].plot(times, np.mean(y2x_sh_per[i],  axis=0),  c='k', ls=':', linewidth = 0.8)
    axes[i].set_ylim(0, 0.7)
    axes[i].set_ylabel('GC (log-variance)')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_title(name_lab[i])
    axes[i].grid()

    #ax_TS.scatter([times[gc_x2y.argmax()]], [5+i*0.5], c='grey', marker = marker[i])
    #ax_TS.scatter([times[gc_y2x.argmax()]], [4.8-i*0.5], c='k',marker = marker[i])
    #axes[i].scatter([times[gc_x2y.argmax()]], np.max(gc_x2y), c='grey', marker = marker[i])
    #axes[i].scatter([times[gc_y2x.argmax()]], np.max(gc_y2x), c='k',marker = marker[i])
    

leg = [Line2D([0], [0], linestyle='None', marker='s', markersize=10,
           markerfacecolor='grey', markeredgecolor='grey', label=f'{lab[0]} → {lab[1]}'), 
      Line2D([0], [0], linestyle='None', marker='s', markersize=10,
           markerfacecolor='k', markeredgecolor='k', label=f'{lab[1]} → {lab[0]}'), 
      Line2D([0], [0], linestyle=':', color='k', label=f'95 percentil shuffled time-points'),
      Line2D([0], [0], linestyle='--', linewidth = 1, color='k', label=f'Mean±std shuffled time-point')]

leg2 = [  Line2D([0], [0], linestyle='None', marker='x', markersize=5,
           markerfacecolor='k', markeredgecolor='k', label=f'Max GC Full'), 
        Line2D([0], [0], linestyle='None', marker='o', markersize=5,
           markerfacecolor='k', markeredgecolor='k', label=f'Max GC Old'), 
        Line2D([0], [0], linestyle='None', marker='^', markersize=5,
           markerfacecolor='k', markeredgecolor='k', label=f'Max GC New'), 
        Line2D([0], [0], linestyle='-', color=col_pc[0], label=lab[0]),
        Line2D([0], [0], linestyle='-', color=col_pc[2], label=lab[1]),
           ]

ax_TS.legend(handles= leg2)
axes[-1].legend(handles= leg, bbox_to_anchor=(-0.75, -0.25), loc='center', ncol=4)


if not os.path.exists(OUT_PATH + f'/Granger/{band}/') :
    os.makedirs(OUT_PATH + f'/Granger/{band}/')
                  
fig.savefig(OUT_PATH + f'/Granger/{band}/{lab[0]}_{lab[1]}_win{window_len}_lag{maxlag}.png')

