
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import json 
import seaborn as sns 
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
#scripts_path = os.path('/scripts')
#from sys import path; path.append('scripts')

from utils import OUT_PATH, FREQ_BAND, ExcludSubj, GetInfo, DataTransformationM1
from scipy.stats import spearmanr

tfr_path = OUT_PATH+ '/Data_longWOBS'
subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
subj_included = ExcludSubj(subj_included, data_path=tfr_path)
print(f"Subject Nbr is {len(subj_included)}")


path = tfr_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time_epoch = d['time_epoch']
    time_tfr=d['time_tfr']
     
coord, areas, elect_list, subj_list, regions = GetInfo(subj_included, data_path=tfr_path)
region = regions
plt.style.use('seaborn-v0_8-dark')

event_idx = {0: 'Old', 1:'New'}
color_event = {0: 'b', 1:'r'}
interesting_ev = [0, 1]

method_pca = 'concat'
data_aug_method = 'mean'
nb_compo=5
fig, axs= plt.subplots(nb_compo,len(FREQ_BAND)+1, figsize=(18, 6))
fig.suptitle('Stability accross trials')
for j in range(nb_compo) : 
    axs[j][0].set_ylabel('PC' + str(j+1))

# share y
ymin, ymax = -11, -8
for ax in axs[:, -1:].flatten():   # not the last columns all the row
    ax.set_ylim(ymin, ymax)

dict_corr = {'New': [], 'Old': []}

df_X_transformed = pd.read_csv(f'{OUT_PATH}/grpPCA/supsubj_{method_pca}/grp_{method_pca}_Xtrans_PCA{nb_compo}.csv').drop(columns = ['Unnamed: 0'])
for ib, band in enumerate(FREQ_BAND) :
    for PC_use in range(nb_compo)  :
        ax = axs[PC_use][ib]
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, 
                                                                      method_pca=method_pca, 
                                                                      nb_compo=PC_use+1, 
                                                                      data_aug_method=data_aug_method, 
                                                                      subj_included=subj_included, 
                                                                      PC_use=PC_use, 
                                                                      data_path=tfr_path)        
        
        X = np.concat([X_train, X_test])
        y = np.concat([y_train, y_test])
        id_old = np.where(y ==1)[0]
        id_new = np.where(y ==2)[0]
        # add copo time serie
        c = 'compo' + str(PC_use + 1)
        time_serie = df_X_transformed.query('freq == @band and compo == @c').drop(columns = [ 'freq' , 'compo', 'subj', 'expl_var']).values
        
        for i_ev,index_  in enumerate([id_old, id_new]) : 
            ax.plot(time_tfr, X[index_, :].mean(0), label = event_idx[i_ev], c = color_event[i_ev], ls='--', alpha = 0.6)
            ax.fill_between(time_tfr, X[index_, :].mean(0) - X[index_, :].std(0), X[index_, :].mean(0) + X[index_, :].std(0), alpha  = 0.2, color = color_event[i_ev]) 
            
            if i_ev == 0 :
                ax.plot(time_tfr, time_serie[0, :int(time_serie.shape[1]/2)], label = 'Mean Grp level', c = color_event[i_ev], alpha = 1, ls='-')
            elif i_ev == 1 : 
                ax.plot(time_tfr, time_serie[0, int(time_serie.shape[1]/2):], label = 'Mean Grp level', c = color_event[i_ev], alpha = 1, ls='-')
            
            # compute the corr coef
            corr =np.mean(spearmanr(X[index_, :].T).statistic)
            dict_corr[event_idx[i_ev]].append(corr)

            #ax.text(s=f'{event_idx[i_ev]} mean r : {np.round(corr, 2)}', y = -10 - (i_ev)*0.7, x = -1)

        ax.grid()  
        axs[-1][ib].set_xlabel('Time (s)')
    axs[0][ib].set_title(band.replace('_', ' ').capitalize())      

## add bb at the end 
for PC_use in range(nb_compo)  :
    ax = axs[PC_use][-1]
    X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= 'broadband', 
                                                                                      method_pca=method_pca, 
                                                                                      data_aug_method=data_aug_method, 
                                                                                      subj_included=subj_included, 
                                                                                      PC_use=PC_use, 
                                                                                      data_path=tfr_path,
                                                                                      nb_compo=PC_use +1, 
                                                                                      pol_cor=(False, 0), 
                                                                                      method ='pca')        
    X = np.concat([X_train, X_test])
    y = np.concat([y_train, y_test])
    id_old = np.where(y ==1)[0]
    id_new = np.where(y ==2)[0]

    # Get the group
    df_time_serie = pd.read_csv(OUT_PATH + f'/grpPCA/supsubj_bb_{method_pca}/grp_Xtrans_PCA5.csv')
    for col in ['Unnamed: 0', 'subj', 'compo', 'freq', 'error'] : 
        if col in df_time_serie.columns :
            df_time_serie = df_time_serie.drop(columns = col)
    time_serie = df_time_serie.values

    for i_ev,index_  in enumerate([id_old, id_new]) : 
        ax.plot(time_epoch, X[index_, :].mean(0), label = f'Trial level: {event_idx[i_ev]}' , c = color_event[i_ev], ls='--', alpha = 0.6, linewidth=1)
        ax.fill_between(time_epoch, X[index_, :].mean(0) - X[index_, :].std(0), X[index_, :].mean(0) + X[index_, :].std(0), alpha  = 0.2, color = color_event[i_ev]) 

        if i_ev == 0 :
            ax.plot(time_epoch, time_serie[PC_use, :int(time_serie.shape[1]/2)], label = f'Mean level: {event_idx[i_ev]}', c = color_event[i_ev], alpha = 1, ls='-')
        elif i_ev == 1 : 
            ax.plot(time_epoch, time_serie[PC_use,  int(time_serie.shape[1]/2):], label = f'Mean level: {event_idx[i_ev]}', c = color_event[i_ev], alpha = 1, ls='-')
        
        corr =np.mean(spearmanr(X[index_, :].T).statistic)
        dict_corr[event_idx[i_ev]].append(corr)

        #ax.text(s=f'{event_idx[i_ev]} mean r : {np.round(corr, 2)}', y = -0.015 -i_ev*0.003, x = -0.6)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    
    ax.grid()  
axs[-1][-1].set_xlabel('Time (s)')
axs[0][-1].set_title('Broadband')  
axs[0][-1].legend(bbox_to_anchor = (1, 1))   

plt.savefig(OUT_PATH + '/final/figG.png')
pd.DataFrame(dict_corr).to_csv(OUT_PATH + '/final/corrTrials.csv')