import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import json 

import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from nilearn import plotting 

plt.style.use('petroff10')


from src.config import OUT_PATH, FREQ_BAND, COL_REG, cmap_pcs, col_pc
from src.setting import ExcludSubj, GetInfo, get_data_grp
from src.decoding import cleandfdecodingTS

dataset_label = 'mf70-160'
tfr_path = OUT_PATH+ '/Data_longWOBS_' + dataset_label

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

# per pc with topomap
data_aug_method = 'duplicat'
method_pca = 'concat'
band = 'high_gamma'
time = time_tfr
data = data_grp[:, :, FREQ_BAND.index(band),:]

saving_dir = OUT_PATH + f'/moving_figs/{band}'

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

compo = pd.read_csv(OUT_PATH + f'/grpPCA_{dataset_label}/supsubj_{method_pca}/grp_{method_pca}_Compo_PCA5.csv').drop(columns = 'Unnamed: 0').query('freq == @band')
ts = pd.read_csv(OUT_PATH + f'/grpPCA_{dataset_label}/supsubj_{method_pca}/grp_{method_pca}_Xtrans_PCA5.csv').drop(columns = ['Unnamed: 0']).query('freq == @band')
for col in ['expl_var', 'compo', 'freq', 'subj'] : 
    if col in compo.columns : 
        compo = compo.drop(columns = col)

    if col in ts.columns : 
        ts = ts.drop(columns = col)

ts = ts.values
compo = compo.values
ts_pc=np.concat([ts[None, :, :int(ts.shape[1]/2)], ts[None, :, int(ts.shape[1]/2):]], axis = 0)
for pc in [0, 1, 2]:  
  act= data.mean(0) * compo[pc, :][:, None]
  max_act = np.max(abs(act))
  min_act =-max_act
  thr= max_act*0.05

  for tp in range(len(time_tfr)) : 
      fig = plt.figure(figsize=(12, 8))
      gs = gridspec.GridSpec(2, 1, height_ratios=[3/5, 2/5], hspace=0.5, wspace=0.3)
      ax_TS = fig.add_subplot(gs[0, 0])  
      ax_topo = fig.add_subplot(gs[1, 0])
      
      ax_TS.plot(time, ts_pc[0, pc, :].T, c=col_pc[pc], ls=':', linewidth = 1, label='Old')
      ax_TS.plot(time, ts_pc[1, pc, :].T, c=col_pc[pc], ls='--', linewidth = 1, label = 'New')
      ax_TS.grid()
      ax_TS.set_xlabel('Time (s)')
      ax_TS.set_title(f'PC {pc+1}')
      ax_TS.legend( bbox_to_anchor=(1, -0.1), ncol=2)

      ax_TS.scatter([time[tp]], ts_pc[0, pc, tp].T, s=20, marker='o', c=['k'], label = 'Old')
      ax_TS.scatter([time[tp]], ts_pc[1, pc, tp].T, s=20, marker='x', c=['k'], label = 'New')
      ax_TS.legend()
      index_thr = np.where(abs(act[:, tp]) > thr)[0]
      n_nodes = len(coord)
      node_sizes = np.full(n_nodes, 0)        
      node_sizes[index_thr] = 20    

      fig = plotting.plot_markers(node_coords = np.array(coord),  
                                  node_size=node_sizes, 
                                  node_values=act[:, tp],
                                  node_cmap='seismic',
                                    display_mode='ortho', 
                                    node_vmin=min_act, 
                                    node_vmax=max_act, 
                                    axes = ax_topo)


      fig.savefig(saving_dir + f'/{pc}_topo_gg_plot_{tp}.png')
      plt.close()
      