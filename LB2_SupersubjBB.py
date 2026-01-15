import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import gaussian_filter1d

from utils import OUT_PATH
from utils import PlotCompoIndividual, BbEvents, GetInfo, CompoThr, ExcludSubj, PolarityCor, computeLagPeak, crosscorr, ConcatPCA, DataTransformationM1

epoch_path = OUT_PATH + '/Data_shortWOBS'
epoch_path = 'outs/Data_shortWOBS'

subj_included = [file.replace('_epochs.p', '') for file in os.listdir(epoch_path) if file[-len('epochs.p'):] == 'epochs.p']
subj_included = ExcludSubj(subj_included,  data_path=epoch_path)
print('Number of subject is ', len(subj_included))

#coord, areas, elects, subj = GetInfo(subj_included, data_path=epoch_path)
coords_reg= pd.read_csv('outs/grpPCA/supsubj_concat/high_gamma_labels.csv')
# get the region only for locals
coord = []
for s in subj_included :
    coord_file = pd.read_csv(f'{epoch_path}/{s}_coords.csv')
    coord.extend(np.vstack([coord_file['x'].values,  coord_file['y'].values, coord_file['z'].values]).T)
coord_array = np.array(coord)
regions = coords_reg['region'] 

event_idx = {0: 'Old', 1:'New'}
color_event = {0: 'b', 1:'r'}
interesting_ev= [0,1]

# Get time 
path = epoch_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time = d['time_epoch']

# polarity correction
pol_cor = True
cor_thr = 0.7

########## PCA computation ########### 
nb_compo = 3
method_pca = 'mean' # 'concat
band = 'broadband'
data_aug_method = 'mean'
method = 'pca'

# 0. Get the data
data = []
subj_list = []
for subj in subj_included: # go over the subject and comptute mean per condi over trials
    data_mean = BbEvents(subj, data_path=epoch_path)
    subj_list.extend([subj] * data_mean.shape[1])
    info_file = f'{epoch_path}/{subj}_info.json'
    data.append(data_mean)
    
data_grp = np.hstack(data)
if pol_cor:
    data_grp = PolarityCor(TFRtr=data_grp, 
                           data_path=epoch_path, 
                           method_pca=method_pca, 
                           subj_included=subj_included)

# 1. Compute for pca method and get a plot of the componants
if method_pca == 'mean' :
    data_grp_mean = data_grp.mean(0)
else : 
    data_grp_mean = np.concat([data_grp[i, :, :] for i in [0, 1]], axis =1)

pca = PCA(nb_compo)
pca.fit(data_grp_mean.T)

df_Compo = pd.DataFrame(pca.components_)
df_Compo['compo'] = ['compo'  +str(i+1) for i in range(nb_compo)]
df_Compo['freq'] = ['broadband']*nb_compo 
ev = pca.explained_variance_ratio_
data_transfrom = np.zeros((2, nb_compo, data_grp.shape[2]))
data_transfrom[0, :, :] = pca.transform(data_grp[0, :,:].T).T
data_transfrom[1, :, :] = pca.transform(data_grp[1, :,:].T).T

fig, ax = plt.subplots(1, 3, figsize = (14, 4), sharey=True)
fig.suptitle(f'Group PCs (Broadband)')

for i in range(3) : 
    ax[i].plot(time, data_transfrom[0, i,:], c = 'blue', label='old/correct', alpha = 1, ls=':')
    ax[i].plot(time, data_transfrom[1, i,:], c = 'red', label='new/correct', alpha = 1, ls=':')
    ax[i].set_title(f'PC {str(i+1)} --  ev : {np.round(ev[i], 2)}')
ax[i].legend(bbox_to_anchor=(1, 1))
plt.show()

PlotCompoIndividual('grp', 
                    df_Componants={'grp' : df_Compo}, 
                    nb_compo = nb_compo,
                    freq_band=['broadband'], 
                    out_path= OUT_PATH + '/grpPCA/supsubj_bb_mean/', 
                    data_path=epoch_path, 
                    show=False)

fig, ax = plt.subplots(1,nb_compo, figsize = (14, 4), sharey=True)
fig.suptitle(f'Group PCs (Broadband)')

for i in range(nb_compo) : 
    ax[i].plot(time, data_transfrom[0, i,:], c = 'blue', label='old/correct', alpha = 0.7)
    ax[i].plot(time, data_transfrom[1, i,:], c = 'red', label='new/correct', alpha = 0.7)
    ax[i].set_title(f'PC {str(i+1)} --  ev : {np.round(ev[i], 2)}')
ax[i].legend()
plt.savefig(OUT_PATH + f'/grpPCA/supsubj_bb_{method_pca}/grp_TimeSeries.png')
plt.close()

# Save
df_X_transform = pd.DataFrame(np.concat([data_transfrom[0, :, :], data_transfrom[1, :, :]], axis=1))
df_X_transform['freq'] = 'broadband'
df_X_transform['subj'] = 'grp'
df_X_transform['compo'] = ['compo' + str(i+1) for i in range(nb_compo)]

df_X_transform.to_csv(f'{OUT_PATH}/grpPCA/supsubj_bb_{method_pca}/grp_Xtrans_PCA{nb_compo}.csv')
df_Compo.to_csv(f'{OUT_PATH}/grpPCA/supsubj_bb_{method_pca}/grp_Compo_PCA{nb_compo}.csv')

########## NMF computation ########### 
method_pca = 'mean'
nb_compo = 4

if method_pca =='mean' :
    data_grp_mean = data_grp.mean(0)
else : 
    data_grp_concat = np.concat([data_grp[0, :,:], data_grp[1, :,:]], axis=1)

df_Componants, df_X_transformed = ConcatPCA({'grp' : data_grp_mean}, 
                                            ch_id = False, 
                                            nb_compo =nb_compo, 
                                            freq_band=['broadband'], 
                                            method ='nmf')

weights=df_Componants['grp'].drop(columns= ['compo', 'freq']).values #(4, 2576) compo, electrods

data_transfrom = np.zeros((2, nb_compo, data_grp.shape[2]))
data_transfrom[0, :, :] = weights @ data_grp[0, :,:]
data_transfrom[1, :, :] = weights @ data_grp[1, :,:]

df_transform = pd.DataFrame(np.concat([data_transfrom[0, :, :], data_transfrom[1, :, :]], axis = 1))
df_transform['subj'] = 'grp'
df_transform['freq']='broadband'
df_transform['compo'] = ['compo' + str(i) for i in range(nb_compo)]

df_Componants['grp'].to_csv(f'{OUT_PATH}/grpNMF/supsubj_bb_{method_pca}/grp_Compo_NMF{nb_compo}.csv')
df_transform.to_csv(f'{OUT_PATH}/grpNMF/supsubj_bb{method_pca}/grp_Xtrans_NMF{nb_compo}.csv')

########## Compare methods ########### 

# Exp var
data_grp_mean = data_grp.mean(0)
data_grp_concat = np.concat([data_grp[i, :, :] for i in [0, 1]], axis =1)

colors = plt.get_cmap('tab10')
pca = PCA(10)
pca.fit(data_grp_mean.T)

fig, ax= plt.subplots(figsize=(10,4))
ax.plot(['PC'+ str(i) for i in np.arange(10)+1], pca.explained_variance_ratio_*100, c='black', label='Mean')

pca.fit(data_grp_concat.T)
ax.plot(['PC'+ str(i) for i in np.arange(10)+1], pca.explained_variance_ratio_*100, c='grey', label='Concatenation')
ax.grid()
ax.set_title('Explained variance')
ax.set_xlabel('Principal components')
ax.set_ylabel('Percent of variance explained (%)')
ax.legend(title = 'BroadBand time-domain PCA', bbox_to_anchor=(1.4, 1))
plt.show()

# Plot mean vs concat
method = 'pca' #nmf
nb_compo = 3

df_X_transformed_concat =pd.read_csv(OUT_PATH + f'/grp{method.upper()}/supsubj_bb_concat/grp_Xtrans_{method.upper()}{nb_compo}.csv').drop(columns = ['Unnamed: 0', 'error'])
df_Componants_mean = pd.read_csv(OUT_PATH + f'/grp{method.upper()}/supsubj_bb_mean/grp_Compo_{method.upper()}{nb_compo}.csv').drop(columns = ['Unnamed: 0'])

fig, ax = plt.subplots(1,nb_compo, figsize = (3*nb_compo, 3), sharey=True)
data = df_X_transformed_concat.query('freq == "broadband"').drop(columns = ['compo', 'freq', 'subj']).values
weights = df_Componants_mean.query('freq == "broadband"').drop(columns = ['compo', 'freq']).values[:3, :]
data1 = weights @ data_grp[0,:, :]
data2 = weights @ data_grp[1,:, :]

for i in range(nb_compo) : 
    ax[i].plot(time, data[i, :int(data.shape[1]/2)], label = f'{event_idx[interesting_ev[0]]} -- Concatenation', ls = '--', c = color_event[interesting_ev[0]])
    ax[i].plot(time, data[i, int(data.shape[1]/2):], label = f'{event_idx[interesting_ev[1]]} -- Concatenation', ls = '--', c= color_event[interesting_ev[1]])
    ax[i].set_ylabel('PC ' + str(i+1))
    ax[i].set_xlabel('Time')
    ax[i].plot(time, data1[i,:], label = f'{event_idx[interesting_ev[0]]} -- Mean', ls = ':', c=color_event[interesting_ev[0]])
    ax[i].plot(time, data2[i, :], label = f'{event_idx[interesting_ev[1]]} -- Mean', ls = ':', c=color_event[interesting_ev[1]])
    ax[i].grid()
ax[-1].legend(bbox_to_anchor = (1, 1))

# Stability across weights 
mean_compo=pd.read_csv(f'{OUT_PATH}/grp{method.upper()}/supsubj_bb_mean/grp_Compo_{method.upper()}{nb_compo}.csv').drop(columns = ['Unnamed: 0', 'compo', 'freq']).values
concat_compo=pd.read_csv(f'{OUT_PATH}/grp{method.upper()}/supsubj_bb_concat/grp_Compo_{method.upper()}{nb_compo}.csv').drop(columns = ['Unnamed: 0', 'compo', 'freq']).values

R = np.zeros((mean_compo.shape[0], concat_compo.shape[0]))
P = np.zeros((mean_compo.shape[0], concat_compo.shape[0]))

for i in range(mean_compo.shape[0]):
    for j in range(concat_compo.shape[0]):
        r, pval = spearmanr(mean_compo[i, :], concat_compo[j, :])
        R[i, j] = r
        P[i, j] = pval

fig, ax = plt.subplots(figsize = (8, 6))
sns.heatmap(abs(R), vmin=0, vmax=1, cmap='Blues', ax=ax, annot=True , fmt='.3f')
ax.set_title('Spearman correlation between mean and concat methods weights')
plt.show()

## Plot stability accorss trials 
fig, axs= plt.subplots(1,3 , sharey=True, figsize=(15, 4))
fig.suptitle('Stability accross trials')
axs[0].set_ylabel('PC1')
axs[1].set_ylabel('PC2')

color_event = {0: 'b', 1:'r'}
event_idx = {0: 'Old', 1:'New'}

for PC_use in [0, 1, 2]  :
    ax = axs[PC_use]
    X_train, y_train, X_test, y_test, True_trials, pca_weights = DataTransformationM1(freq= band, 
                                                                                      method_pca=method_pca, 
                                                                                      data_aug_method=data_aug_method, 
                                                                                      subj_included=subj_included, 
                                                                                      PC_use=PC_use, 
                                                                                      data_path=epoch_path, 
                                                                                      pol_cor=pol_cor, 
                                                                                      method =method)        
    X = np.concat([X_train, X_test])
    y = np.concat([y_train, y_test])
    id_old = np.where(y ==1)[0]
    id_new = np.where(y ==2)[0]

    c = 'compo' + str(PC_use + 1)

    # Get the group
    df_time_serie = pd.read_csv(OUT_PATH + f'/grp{method.upper()}/supsubj_bb_{method_pca}/grp_Xtrans_{method.upper()}{nb_compo}.csv')
    for col in ['Unnamed: 0', 'subj', 'compo', 'freq', 'error'] : 
        if col in df_time_serie.columns :
            df_time_serie = df_time_serie.drop(columns = col)
    time_serie = df_time_serie.values

    for i_ev,index_  in enumerate([id_old, id_new]) : 
        ax.plot(time, X[index_, :].mean(0), label = f'Trial level: {event_idx[i_ev]}' , c = color_event[i_ev], ls=':', alpha = 0.6, linewidth=1)
        ax.fill_between(time, X[index_, :].mean(0) - X[index_, :].std(0), X[index_, :].mean(0) + X[index_, :].std(0), alpha  = 0.2, color = color_event[i_ev]) 

        if i_ev == 0 :
            ax.plot(time, time_serie[PC_use, :int(time_serie.shape[1]/2)], label = f'Mean level: {event_idx[i_ev]}', c = color_event[i_ev], alpha = 1, ls='-')
        elif i_ev == 1 : 
            ax.plot(time, time_serie[PC_use,  int(time_serie.shape[1]/2):], label = f'Mean level: {event_idx[i_ev]}', c = color_event[i_ev], alpha = 1, ls='-')
        
        corr =np.mean(spearmanr(X[index_, :].T).statistic)
        ax.text(s=f'{event_idx[i_ev]} mean r : {np.round(corr, 2)}', y = -0.015 -i_ev*0.001, x = -0.6)
    
    ax.grid()  
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PC ' + str(PC_use+1))
    if PC_use == 0 :
        ax.legend()      


# Lags between PCs at peak level 
lag_dict={'mean_old':[], 'mean_new':[], 'concat_old':[], 'concat_new':[]}

for pca_method in ['mean', 'concat'] :
    data= pd.read_csv(OUT_PATH + f'/grpPCA/supsubj_bb_{pca_method}/grp_Xtrans_PCA3.csv').query('freq == "broadband"').drop(columns = ['Unnamed: 0', 'freq', 'compo', 'subj']).values #, 'expl_var']).values
    data = np.concat([[data[:, :int(data.shape[1]/2)]], [data[:, int(data.shape[1]/2):]]], axis=0)
    for i_condi, condi in enumerate(['old', 'new']) : 
        coni = pd.DataFrame(data[i_condi, :,:].T).T

        if condi == 'new' :
            win_high = {0 : [([100, 200], 0.0070), ([200, 400], 0.0090), ([400, 600], 0.009)], 
            1: [([100, 130], 0.0075), ([200, 400], -1), ([400, 450], -0.0045)]}
        elif condi == 'old' :
            win_high = {0 : [([100, 200], 0.0075), ([200, 400], 0.0090), ([400, 600], 0.0065)], 
            1: [([100, 130], 0.0075), ([200, 400], -1), ([400, 600], -0.0045)]}

        lag = computeLagPeak(coni.values, time, win_high, method_pca=pca_method, show=True)
        lag_dict[f'{pca_method}_{condi}'].extend(lag)

y= np.linspace(0, 4, 5)
for key, values in lag_dict.items():
    y += 0.1
    plt.scatter(x=y, y=values, label=key, s=20)
plt.ylabel('Lag (s)')
plt.xlabel('Peak pairs PC1-PC2')
plt.legend()
plt.grid()
plt.show()

# Coherence : from subj level to grp level 
from utils import subject_variance_explained, loso_pca_stability, intersubject_correlation
concat_data_list = []
subj_list = []

for subj in subj_included : 
    epochs = BbEvents(subj, data_path=epoch_path)
    subj_list.extend([subj] * epochs.shape[1])

if method_pca == 'mean':
    data_grp_ = data_grp.mean(0)
else : 
    data_grp_ = np.concat([data_grp[i, :, :] for i in [0, 1]], axis =1)

pca = PCA(nb_compo)
pca.fit(data_grp_.T)
W = pca.components_

r2 = subject_variance_explained(data_grp_, subj_list, W.T)
print(r2.mean()*100)

stability = loso_pca_stability(data_grp_, subj_list, 3)
print("Mean LOSO stability:", stability.mean(axis=0))

for k in range(3):
    mean_r, all_r = intersubject_correlation(data_grp_, subj_list, W, k)
    print(f'PC{k+1}: Mean inter-subject correlation: {mean_r}') 

# Coherence time serie correlation 
grp_TS=pd.read_csv(f'{OUT_PATH}/grpPCA/supsubj_bb_{method_pca}/grp_Xtrans_PCA{nb_compo}.csv').drop(columns = 'Unnamed: 0')
subj_TS=pd.read_csv(f'{OUT_PATH}/subjPCA/Broadband/subj_Xtrans_PCA{nb_compo}_{method_pca}.csv').drop(columns = 'Unnamed: 0')
index_col = grp_TS.columns[-4]
filter_ = True

fig, ax = plt.subplots(len(subj_included),nb_compo, figsize = (15, len(subj_included)*2))
compos_count = {'PC'+str(i):0 for i in range(nb_compo)}

for subj_i, subj in enumerate(subj_included) : 
    subj_df = subj_TS.query('subj == @subj') 
    for j in range(nb_compo) : 
        ax[subj_i][j].plot(subj_df.set_index('compo').loc[f'compo{j+1}', :index_col].values)
        ax[subj_i][j].set_xticklabels([])

    data1 = grp_TS.drop(columns = ['subj', 'freq', 'compo']).values.T
    data2 = subj_df.drop(columns = ['subj', 'freq', 'compo']).values.T

    if filter_ :
        data1 = gaussian_filter1d(data1.T, sigma=20).T
        data2 = gaussian_filter1d(data2.T, sigma=20).T

    R = np.zeros((data1.shape[1], data2.shape[1]))
    P = np.zeros((data1.shape[1], data2.shape[1]))

    for i in range(data1.shape[1]):
        for j in range(data2.shape[1]):
            r, pval = spearmanr(data1[:, i], data2[:, j])
            R[i, j] = r
            P[i, j] = pval

    P_corr = P * nb_compo*nb_compo # Bonferroni correction for 3x3 tests

    xcor, ycor = np.where(R > 0.8)
    for x, y in zip(xcor, ycor) :
        compos_count['PC'+str(x)] +=1 
        ax[subj_i][y].plot(grp_TS.set_index('compo').loc['compo' + str(x+1), :index_col].values, color = 'red')
        ax[subj_i][y].set_xlabel(f'r : {np.round(R[x, y], 2)}, p-value bh-corr: {P_corr[x, y]}')
    
    xcor, ycor = np.where(R < - 0.8)
    for x, y in zip(xcor, ycor) :
        compos_count['PC'+str(x)] +=1 
        ax[subj_i][y].plot(- grp_TS.set_index('compo').loc['compo' + str(x+1), :index_col].values, color = 'red')
        ax[subj_i][y].set_xlabel(f'r : {np.round(R[x, y], 2)}, p-value bh-corr: {P_corr[x, y]}')

    ax[subj_i][0].set_ylabel(subj) 


