import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import json 

import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

plt.style.use('seaborn-v0_8-dark')

from src.config import OUT_PATH, FREQ_BAND, COL_REG, cmap_pcs, col_pc
from src.setting import ExcludSubj, GetInfo, get_data_grp
from src.decoding import cleandfdecodingTS


tfr_path = OUT_PATH+ '/Data_longWOBS'
subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
subj_included = ExcludSubj(subj_included, data_path=tfr_path)
print(f"Subject Nbr is {len(subj_included)}")

path = tfr_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time_epoch = d['time_epoch']
    time_tfr=d['time_tfr']
     
coord, _, _, _, region = GetInfo(subj_included, data_path=tfr_path)
data_grp = get_data_grp(subj_included, type_data='tfr', data_path=tfr_path)

data_aug_method = 'mean'
method_pca = 'concat'
band = 'high_gamma'
model_name = 'LR'
time = time_tfr
PCS =[0, 1, 2]

data = data_grp[:, :, FREQ_BAND.index(band), : ]

compo = pd.read_csv(OUT_PATH + f'/grpPCA/supsubj_{method_pca}/grp_{method_pca}_Compo_PCA5.csv').drop(columns = 'Unnamed: 0').query('freq == @band')
ts = pd.read_csv(OUT_PATH + f'/grpPCA/supsubj_{method_pca}/grp_{method_pca}_Xtrans_PCA5.csv').drop(columns = ['Unnamed: 0']).query('freq == @band')
for col in ['expl_var', 'compo', 'freq', 'subj'] : 
    if col in compo.columns : 
        compo = compo.drop(columns = col)

    if col in ts.columns : 
        ts = ts.drop(columns = col)

ts = ts.values
compo = compo.values
ts_pc=np.concat([ts[None, PCS, :int(ts.shape[1]/2)], ts[None, PCS, int(ts.shape[1]/2):]], axis = 0)

for tp in range(len(time)) : 
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 4, height_ratios=[3/5, 2/5], hspace=0.5, wspace=0.3)
    ax_TS = fig.add_subplot(gs[0, :])  
    ax_w1 = fig.add_subplot(gs[1, 0], polar=True)
    ax_w2 = fig.add_subplot(gs[1, 1], polar=True)
    ax_w3 = fig.add_subplot(gs[1, 2], polar=True)
    ax_w4 = fig.add_subplot(gs[1, 3], polar=True)

    # Weights of the decoding model as a gradiant
    weightLR = []
    for pc_use in PCS : 
        df_final = pd.read_csv(OUT_PATH + f'/Decoding/{band}/fullmodel/{method_pca}_{data_aug_method}_{pc_use}_{model_name}full.csv')
        result = cleandfdecodingTS(df_final, len(time))
        weight_mean= abs(result['weight_mean'])
        weight_sh_mean= abs(result['weight_sh_mean'])
        weight_sh_std= result['weight_sh_std']
        weight_prop = weight_mean/(weight_sh_mean+weight_sh_std)
        weightLR.append(weight_prop)

    for i in range(len(time) - 1):
        for pc in PCS : 
            ax_TS.fill_between(time[i:i+2],ts_pc[0, pc, i:i+2],ts_pc[1, pc, i:i+2],color=cmap_pcs[pc](weightLR[pc][i]),linewidth=0)
    
    for pc_use in PCS : 
        ax_TS.plot(time, ts_pc[0, pc_use, :].T, c=col_pc[pc_use], ls=':', linewidth = 1)
        ax_TS.plot(time, ts_pc[1, pc_use, :].T, c=col_pc[pc_use], ls='--', linewidth = 1)
        
    ax_TS.grid()
    ax_TS.set_xlabel('Time (s)')
    ax_TS.set_title('PCs TIME SERIES')
    ax_TS.legend( bbox_to_anchor=(1, -0.1), ncol=2)

    ax_TS.scatter([time[tp], time[tp], time[tp]], ts_pc[0, :, tp].T, s=50, marker='.', c=['b', 'b', 'b'], label = 'Old')
    ax_TS.scatter([time[tp], time[tp], time[tp]], ts_pc[1, :, tp].T, s=50, marker='.', c=['r', 'r', 'r'], label = 'New')

    n_regions = len(COL_REG.keys())
    angles = np.linspace(0, 2 * np.pi, n_regions, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    # pos and neg sum /count 
    for PC_use in PCS : 
        act= data.mean(0)[:, tp] * compo[PC_use, :][:, None]
        df_act = pd.DataFrame(act)
        df_abs = df_act.copy().apply(lambda x : abs(x))
        df_act['region'] = region
        df_abs['region'] = region
        # mean of the absolute value --> wher is the main contribution
        means_abs = df_abs.groupby('region').mean().loc[:, PC_use].reindex(COL_REG.keys()).values
        means_abs = np.concatenate([means_abs, [means_abs[0]]])
        ax_w1.plot(angles, means_abs*1000, linewidth=1, label=f"PC{PC_use+1}", c = col_pc[PC_use])
        ax_w1.fill(angles, means_abs*1000, alpha=0.15, c = col_pc[PC_use])
        ax_w1.set_title('MEAN OF |WEIGHTS|', y=1.1)

        # largest 
        index_largest = df_abs.nlargest(100, columns = PC_use).index
        largest = df_act.loc[index_largest, :].groupby('region').count().loc[:, PC_use]
        largest = largest.reset_index()
        for reg in df_act['region'] : 
            if reg not in largest['region'].values :
                largest = pd.concat([largest, pd.DataFrame([[reg, 0]], columns = ['region', PC_use])])
        largest = largest.set_index('region').reindex(COL_REG.keys()).values
        largest = np.concatenate([largest, [largest[0]]])
        ax_w2.plot(angles, largest, linewidth=1, label=f"PC{PC_use}", c = col_pc[PC_use])
        ax_w2.fill(angles, largest, alpha=0.15, c = col_pc[PC_use])
        ax_w2.set_title('LARGEST |WEIGHTS| COUNT', y=1.1)

        pos_sum = df_act[df_act[PC_use] > 0][[PC_use, 'region']].groupby('region').sum().reindex(COL_REG.keys()).fillna(0).values
        neg_sum = df_act[df_act[PC_use] < 0][[PC_use, 'region']].groupby('region').sum().reindex(COL_REG.keys()).fillna(0).values
        count = df_act[[PC_use, 'region']].groupby('region').count().reindex(COL_REG.keys()).fillna(0).values

        pos_sum = np.concatenate([pos_sum, [pos_sum[0]]])
        neg_sum = np.concatenate([neg_sum, [neg_sum[0]]])
        count = np.concatenate([count, [count[0]]])

        ax_w3.plot(angles, pos_sum/count, linewidth=1, label=f"PC{PC_use}", c = col_pc[PC_use])
        ax_w3.fill(angles, pos_sum/count, alpha=0.15, c = col_pc[PC_use])
        ax_w3.set_title('(+) WEIGHT SUM/COUNT', y=1.1)

        ax_w4.plot(angles, -neg_sum/count, linewidth=1, label=f"PC{PC_use}", c = col_pc[PC_use],)
        ax_w4.fill(angles, -neg_sum/count, alpha=0.15, c = col_pc[PC_use])
        ax_w4.set_title('(-) WEIGHT SUM/COUNT', y=1.1)
        
        scale = 1000  # change to 1e6 if you want micro-units etc.
        formatter = mticker.FuncFormatter(lambda x, pos: f"{x*scale:g}")
        ax_w3.yaxis.set_major_formatter(formatter)
        ax_w4.yaxis.set_major_formatter(formatter)

    for a in [ax_w1, ax_w2, ax_w3, ax_w4] : 
        a.set_xticks(angles[:-1])
        a.set_xticklabels(COL_REG.keys(), fontsize=8)
        a.set_theta_offset(np.pi / 2)
        a.set_theta_direction(-1)
    ax_w1.set_ylim(0, 5)
    ax_w2.set_ylim(0, 50)
    ax_w3.set_ylim(0, 0.004)
    ax_w4.set_ylim(0, 0.004)


    ax_w1.legend(
            loc='upper left',
            bbox_to_anchor=(-0.1, 1.5), ncol  =3)
    
    fig.savefig(OUT_PATH + f'/final/{band}/gg_plot_{tp}.png')
    plt.close()
    