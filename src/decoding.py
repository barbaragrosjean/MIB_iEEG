import os
import pickle
import json

import numpy as np
import pandas as pd
import re
import random

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, log_loss, hinge_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import GridSearchCV, StratifiedKFold


plt.style.use('seaborn-v0_8-dark')

from src.config import OUT_PATH, EVENT_ID
from src.decomposition import DataTransformationM1, DataTransformationM1Raw
   
    
def CheckTrials(X_train, y_train, event=list(EVENT_ID.keys())[:2] , out_path=OUT_PATH + '/Decoding', label = '', save=False, color_ev = {0 : 'r', 1 : 'b'}, freq='high_gamma', data_path=OUT_PATH + '/Data') : 
    # get time 
    files_info = [file for file in os.listdir(data_path) if file[-len('info.json'):] == 'info.json']
    with open(data_path + f'/{files_info[0]}') as json_data:
        d = json.load(json_data)
        if freq =='broadband' :
            time = d['time_epoch']
        else : 
            time = d['time_tfr']
        json_data.close()

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    fig, axs = plt.subplots(figsize = (5, 3)) 
    fig.suptitle('Mean PC1 over trials', y=1.005)
    fig.tight_layout()
    for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
        axs.plot(time, X_train[ev, :].mean(0), label = event[ev_i], color = color_ev[ev_i])
    axs.legend()
    if save :
        fig.savefig(out_path + f'/{freq}_{label}CheckTrials.png')
        plt.close()
    else : 
        plt.show()

    fig, axs = plt.subplots(1, 2, figsize = (15, 4))
    fig.suptitle(f'PCs trial exemple on event', y= 1.1)

    for ev_i, ev in enumerate([id_ev1, id_ev2]) :
        ids = random.sample(list(ev), 5) 
        for i in ids :
            axs[ev_i].plot(time, X_train[i, :])
            axs[ev_i].set_title(f'Event {event[ev_i]}')

    if save :
        fig.savefig(out_path + f'/{freq}_{label}CheckTrials.png')
        plt.close()
    else :
        plt.show()

def CheckTrialsMean(X_train,X_test, y_train, freq , event=list(EVENT_ID.keys())[:2], color_ev = {0 : 'r', 1 : 'b'}, save = False, out_path =OUT_PATH + '/Decoding', label='', data_path=OUT_PATH + '/Data') :
    files_info = [file for file in os.listdir(data_path) if file[-len('info.json'):] == 'info.json']
    with open(data_path + f'/{files_info[0]}') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']
        json_data.close()
    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]
    fig, axs = plt.subplots(figsize = (15, 4))
    fig.suptitle(f'Train and Test samples')

    for ev_i, ev in enumerate([id_ev1, id_ev2]) :
        axs.plot(time, X_train[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean Train')
        axs.fill_between(time, X_train[ev, :].mean(0) - X_train[ev, :].std(0), X_train[ev, :].mean(0) + X_train[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std Train')
        axs.plot(time, X_test[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Train')
    axs.legend()
    axs.grid()

    if save :
        fig.savefig(out_path + f'/{freq}_{label}CheckTrialsMean.png')
        plt.close()
    else : 
        plt.show()

def LR(band, method_pca, data_aug_method,subj_included, iteration=100, perm=False, PC_use=0, save=False, out_path=f'{OUT_PATH}/Decoding/', iter_perm=1, data_path = OUT_PATH + '/Data') : 
    Y_PRED = []
    Y_PRED_s = []
    Y_TEST = []
    MODELS_weights = []
    MODELS_weights_s = []
    PCA_weights= []
    acc_perm_list =[]
    ll_perm_list=[]

    for i in range(iteration) :   
        X_train, y_train, X_test, y_test, True_trials, pca_weights = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path)        
        X_train, y_train = shuffle(X_train, y_train, random_state =0)

        if i == 0 :
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'linear'] 
            }

            base_model = LogisticRegression(max_iter=1000)
            grid = GridSearchCV(base_model, param_grid, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_

        model = LogisticRegression(**best_params, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        MODELS_weights.append(model.coef_)
    
        Y_PRED.extend(y_pred)
        Y_TEST.extend([1, 2])
        PCA_weights.append(pca_weights[PC_use, :])

        if perm : 
            for j in range(iter_perm) :
                # Shuffle the labels
                y_train_s = shuffle(y_train) #, random_state=0)
                model_s = LogisticRegression(**best_params, max_iter=1000)
                model_s.fit(X_train, y_train_s)    
                y_pred_s = model_s.predict(X_test) 
                
                acc_perm_list.append(accuracy_score(y_pred=y_pred, y_true=y_test))
                ll_perm_list.append(log_loss(y_test, model_s.predict_proba(X_test)[:,1]))
                Y_PRED_s.append(y_pred_s)
                MODELS_weights_s.append(model_s.coef_)

    # save the result
    PCA_weights = np.vstack(PCA_weights)
    MODELS_weights = np.vstack(MODELS_weights)
 
    out_dir = out_path + band
    if not os.path.exists(out_dir) : 
        os.makedirs(out_dir)

    sumsum = pd.DataFrame()
    sumsum.loc['band', 0] = band
    sumsum.loc['method_pca', 0] = method_pca
    sumsum.loc['method_data_augm', 0] = data_aug_method
    sumsum.loc['nb_iter', 0] = iteration
    sumsum.loc['best_param', 0] = best_params
    sumsum.loc['trial_truth', 0] = np.round(True_trials.mean(), 2)
    sumsum.loc['y_pred', 0] = [Y_PRED] 
    sumsum.loc['y_true', 0] = [Y_TEST] 
    sumsum.loc['F1', 0] = np.round(f1_score(Y_PRED, Y_TEST), 3) # mean F1 over the 100
    sumsum.loc['accuracy', 0] = np.round(accuracy_score(Y_PRED, Y_TEST), 3) # mean acc over the 100
    count_unique = np.unique(Y_PRED, return_counts=True) # balanced
    sumsum.loc[f'count', 0] = count_unique
    sumsum.loc['pca_weights', 0] = [PCA_weights]
    sumsum.loc['model_weights', 0] = [MODELS_weights]

    if perm :
        sumsum.loc['nb_iter_perm', 0] = iter_perm
        sumsum.loc['acc_perm_list', 0] = acc_perm_list
        sumsum.loc['ll_perm_list', 0] = ll_perm_list
        sumsum.loc['y_pred_perm', 0] = Y_PRED_s
        sumsum.loc['model_weights_s', 0] = [MODELS_weights_s]


    # info correlation pca stability
    n_runs = PCA_weights.shape[0]
    corr_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(i, n_runs):
            rho, _ = spearmanr(PCA_weights[i], PCA_weights[j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
    corr = pd.DataFrame(np.abs(corr_matrix))
    corr.to_csv(out_dir + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_correlation.csv')
    lower_tri = np.tril(corr.values, k=-1)  
    mean = lower_tri[lower_tri != 0].mean()
    sumsum.loc['mean_corPC', 0] = mean

    if save:
        sumsum.to_csv(out_dir + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_summary.csv')
    else : 
        return sumsum

def TemporalGeneralization(band,method_pca,data_aug_method, subj_included, PC_use=0, undersampling=False, save=False, data_path = OUT_PATH + '/Data') : 
    out_dir = f'{OUT_PATH}/Decoding/{band}'
    if not os.path.exists(out_dir) : 
        os.makedirs(out_dir)

    X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path)      
    
    X = np.concat([X_train, X_test], axis =0)
    y = np.concat([y_train, y_test])

    if undersampling : 
        X = X[:, ::5]

    _, n_time = X.shape
    scores = np.zeros((n_time, n_time))

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for t_train in range(n_time):
        X_t = X[:, t_train].reshape(-1, 1)

        for t_test in range(n_time):
            X_te = X[:, t_test].reshape(-1, 1)
            fold_scores = []

            for train_idx, test_idx in kf.split(X_t):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_t[train_idx], y[train_idx])
                y_pred = clf.predict(X_te[test_idx])
                fold_scores.append(accuracy_score(y[test_idx], y_pred))

            scores[t_train, t_test] = np.mean(fold_scores)

    fig, ax = plt.subplots()
    im = ax.imshow(scores, vmin=0, vmax=1, origin='lower', aspect='auto', cmap='PuOr')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Decoding Accuracy")
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Train Time")
    ax.set_title(f"Temporal Generalization - mean score : {np.round(np.mean(scores), 2)}")
    if save :
        fig.savefig(OUT_PATH + '/Decoding/'+ band + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TemporalGeneralization.png' )
        plt.close()
    else : 
        plt.show()

def TemporalLR(band, method_pca, data_aug_method,subj_included, iteration=100, PC_use=0, save=False, data_path = OUT_PATH + '/Data'):
    Y_PRED = []
    Y_TEST = []
    MODELS_weights = []

    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}
    best_params = {}

    for i in range(iteration) :     
        
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path )        
        X_train, y_train = shuffle(X_train, y_train, random_state =0)

        weights = []
        y_pred= []
        y_test_all = []
        for t_point in range(X_train.shape[1]) :
            if i == 0 : 
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
                grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring='accuracy')
                grid.fit(X_train[:, t_point].reshape(-1, 1), y_train)
                best_params[t_point] = grid.best_params_
                best_param = best_params[t_point]
                
            else : 
                best_param = best_params[t_point]

            clf = LogisticRegression(**best_param).fit(X_train[:,t_point].reshape(-1, 1), y_train)
            weights.append(clf.coef_.ravel())
            y_pred.append(clf.predict(X_test[:,t_point].reshape(-1, 1)))
            y_test_all.append(y_test)

        Y_PRED.append(np.vstack(y_pred))
        Y_TEST.append(np.vstack(y_test_all))

        MODELS_weights.append(np.vstack(weights)) 
    MODELS_weights = np.hstack(MODELS_weights)
    Y_PRED = np.concat(Y_PRED, axis =1)
    Y_TEST = np.concat(Y_TEST,  axis =1)

    # analysis 
    fig, axs = plt.subplots(3, 1, figsize = (10, 8),height_ratios=[0.2, 0.4, 0.4], sharex=True)
    fig.suptitle('Model analysis: Temporal Importance')
    fig.tight_layout()

    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        if band == 'broadband' :
            time = d['time_epoch']
        else :
            time = d['time_tfr']
        json_data.close()
    color_ev = {0 : 'r', 1 : 'b'}
    event= list(EVENT_ID.keys())[:2]

    weights_clf = MODELS_weights.mean(1).ravel()
    axs[1].plot(time,abs(weights_clf),linewidth=1,color='black', label = 'Mean')
    axs[1].fill_between(time,abs(weights_clf - MODELS_weights.std(1).ravel()),abs(weights_clf + MODELS_weights.std(1).ravel()),alpha=0.3, linewidth=0,color='gray', label='Std')
    axs[1].set_title('LR Weights over iterations', size = 10)
    axs[1].grid()
    axs[1].legend()

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
        axs[0].plot(time, X_train[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean over Training')
        axs[0].fill_between(time, X_train[ev, :].mean(0) - X_train[ev, :].std(0), X_train[ev, :].mean(0) + X_train[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std over Training')
        axs[0].plot(time, X_test[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Testing')
    
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Last run example', size=10)

    accuracies = (Y_PRED == Y_TEST).mean(axis=1)
    axs[2].plot(time, accuracies, color='black',linewidth=1, label='Mean')
    axs[2].set_title("Accuracy per Time Point", size = 10)
    axs[2].legend()
    axs[2].set_xlabel("Time")

    #2. summary
    sumsum = pd.DataFrame()

    # info test
    sumsum.loc['band', 0] = band
    sumsum.loc['method_pca', 0] = method_pca
    sumsum.loc['method_data_augm', 0] = data_aug_method
    sumsum.loc['nb_iter', 0] = iteration
    sumsum.loc['acc', 0] = np.mean(accuracies)

    # save
    if save:
        out_dir = f'{OUT_PATH}/Decoding/{band}'
        if not os.path.exists(out_dir) : 
            os.makedirs(out_dir)
        sumsum.to_csv(out_dir+ f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TpointSummary.csv')
        fig.savefig(out_dir+ f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TpointTemporalImportance.png' )
        plt.close()
    else :
        plt.plot()
        return sumsum

def TemporalLRRaw(band, data_aug_method,subj_included=[], iteration=100, PC_use=False, method_pca=False, save=False, data_path = OUT_PATH + '/Data', pol_cor=False, out_path = OUT_PATH + '/Decoding'):
    Y_PRED = []
    Y_TEST = []
    MODELS_weights = []
    best_params = {}
    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']}

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
        subj_included = ExcludSubj(subj_included, data_path=data_path)

    for i in range(iteration) :     
        if PC_use == False : 
            X_train, y_train, X_test, y_test, _ = DataTransformationM1Raw(freq= band, data_aug_method=data_aug_method, subj_included=subj_included, data_path=data_path, pol_cor=pol_cor)        
        else : 
            X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path = data_path, pol_cor=pol_cor )                

        # Shuffle 
        X_train, y_train = shuffle(X_train, y_train, random_state =0)
        weights = []
        y_pred= []
        y_test_all = []
        loss = []
        for t_point in range(X_train.shape[-1]) :
            if i == 0 : 
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
                grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring='accuracy')
                grid.fit(X_train[:,:,t_point], y_train)
                best_params[t_point] = grid.best_params_
                best_param = best_params[t_point]
                
            else : 
                best_param = best_params[t_point]

            clf = LogisticRegression(**best_param).fit(X_train[:,:,t_point], y_train)
            weights.append(clf.coef_.ravel())
            y_pred.append(clf.predict(X_test[:,:,t_point]))
            y_test_all.append(y_test)

        Y_PRED.append(np.vstack(y_pred))
        Y_TEST.append(np.vstack(y_test_all))
        MODELS_weights.append(np.vstack(weights)) 

    MODELS_weights = np.concat([i.reshape(i.shape[0], i.shape[1], 1) for i in MODELS_weights], axis=2)
    Y_PRED = np.concat(Y_PRED, axis =1)
    Y_TEST = np.concat(Y_TEST,  axis =1)

    # analysis 

    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        if band == 'broadband' :
            time = d['time_epoch']
        else :
            time = d['time_tfr']
        json_data.close()

    weights_clf = MODELS_weights.mean(-1)

    color_ev = {0 : 'r', 1 : 'b'}
    event= list(EVENT_ID.keys())[:2]

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    if PC_use == False : 
        X_train_to_plot = X_train.mean(1)
        X_test_to_plot = X_test.mean(1)
        title_label = 'Mean channels'
        l = 'Raw'
        fig, axs = plt.subplots(3, 1, figsize = (10, 8), sharex=False)
        for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
            axs[0].plot(time, X_train_to_plot[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean over Training')
            axs[0].fill_between(time, X_train_to_plot[ev,:].mean(0) - X_train_to_plot[ev, :].std(0), X_train_to_plot[ev, :].mean(0) + X_train_to_plot[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std over Training')
            axs[0].plot(time, X_test_to_plot[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Testing')
        
        axs[0].legend()
        axs[0].grid()
        axs[0].set_title(title_label + ' last run example', size=10)
        axs[-2].set_ylabel('Channels')
    else : 
       
        fig, axs = plt.subplots(4, 1, figsize = (10, 8), sharex=False)
        for pc in PC_use:
            X_train_to_plot = X_train[:, pc, :]
            X_test_to_plot = X_test[:, pc, :]
            title_label = 'PC' + str(pc+1)
            for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
                axs[pc].plot(time, X_train_to_plot[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean over Training')
                axs[pc].fill_between(time, X_train_to_plot[ev,:].mean(0) - X_train_to_plot[ev, :].std(0), X_train_to_plot[ev, :].mean(0) + X_train_to_plot[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std over Training')
                axs[pc].plot(time, X_test_to_plot[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Testing')
            
            axs[pc].legend()
            axs[pc].grid()
            axs[pc].set_title(title_label + ' last run example', size=10)
        axs[-2].set_ylabel('PCs')
        l = method_pca + '_pc' + ''.join([str(item+1) for item in PC_use])

    fig.suptitle('Model analysis: Temporal Importance')
    fig.tight_layout()

    sns.heatmap(abs(weights_clf.T), cmap='Blues', xticklabels=False, yticklabels=False,vmin=0, ax=axs[-2])
    axs[-2].set_title('LR Weights per time point mean over iterations', size = 10)

    accuracies = (Y_PRED == Y_TEST).mean(axis=1)
    std = (Y_PRED == Y_TEST).std(axis=1)
    axs[-1].plot(time, accuracies, color='black',linewidth=1, label='Mean')
    axs[-1].fill_between(time,abs(accuracies - std),abs(accuracies + std),alpha=0.3, linewidth=0,color='gray', label='Std')
    axs[-1].set_title(f"Accuracy per Time Point {l} - mean : {np.round(np.mean(accuracies), 2)}", size = 10)
    axs[-1].legend()
    axs[-1].set_xlabel("Time")
    axs[-1].grid()

    #2. summary
    sumsum = pd.DataFrame()
    # info test
    sumsum['band'] = band
    sumsum['method_data_augm'] = data_aug_method
    sumsum['nb_iter'] = iteration
    sumsum['gg_accuracy'] = np.round(np.mean(accuracies), 2)
    sumsum['acc_mean'] = [accuracies]
    sumsum['acc_std'] = [std]
    sumsum['weights_clf'] =[weights_clf]
    
    # save
    if save :
        out_dir = out_path + f'/{band}'
        if not os.path.exists(out_dir) : 
            os.makedirs(out_dir)
        pd.DataFrame(weights_clf).to_csv(out_dir + f'/{band}_{l}_{data_aug_method}_weights_CLF.csv')
        sumsum.to_csv(out_dir + f'/{band}_{l}_{data_aug_method}_TpointSummary.csv')
        fig.savefig(out_dir + f'/{band}_{l}_{data_aug_method}_TpointTemporalImportance.png' )
        plt.close()
    else : 
        plt.show()
        return sumsum

def TemporalGeneralizationRaw(band, data_aug_method, subj_included, save=False,PC_use=False,method_pca=False, undersampling=False, data_path = OUT_PATH + '/Data') : 
    out_dir = f'{OUT_PATH}/Decoding/{band}'
    if not os.path.exists(out_dir) : 
        os.makedirs(out_dir)

    if PC_use == False : 
        X_train, y_train, X_test, y_test, _ = DataTransformationM1Raw(freq= band, data_aug_method=data_aug_method, subj_included=subj_included, data_path=data_path)        
    else : 
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path)                

    X = np.concat([X_train, X_test], axis =0)
    y = np.concat([y_train, y_test])

    if undersampling : 
        X = X[:, :, ::5]

    #for pc in range(X.shape[1]) :    
        #scaler = StandardScaler()
        #X[:,pc, :] = scaler.fit_transform(X[:, pc, :])

    _, _, n_time = X.shape
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros((n_time, n_time))

    for t_train in range(n_time):
        X_t_train = X[:, :, t_train] 
        for t_test in range(n_time):
            X_t_test = X[:, :, t_test]
            fold_scores = []
            for train_idx, test_idx in kf.split(X_t_train):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_t_train[train_idx], y[train_idx])
                y_pred = clf.predict(X_t_test[test_idx])
                fold_scores.append(accuracy_score(y[test_idx], y_pred))
            scores[t_train, t_test] = np.mean(fold_scores)

    fig, ax = plt.subplots()
    im = ax.imshow(scores, vmin=0, vmax=1, origin='lower', aspect='auto', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Decoding Accuracy")
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Train Time")
    if PC_use == False : 
        l = 'Raw'
    else : 
        l = method_pca + '_pc' + ''.join([str(item+1) for item in PC_use])
    
    ax.set_title(f"Temporal Generalization {l.replace('_', ' ')} -- mean accuracy {np.round(np.mean(scores), 2)}")

    if save :
        fig.savefig(out_dir + f'/{band}_{l}_{data_aug_method}_TemporalGeneralization.png' )
        plt.close()
    else : 
        plt.show()

def CompareClassifier(band,method_pca,data_aug_method,subj_included, nb_iter=100, PC_use=0, save=False, perm=False, out_path = OUT_PATH + '/Decoding', data_path=OUT_PATH + '/Data', pol_cor=False) :

    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'SVC_linear': SVC(kernel='linear', probability=True),
        'SVC_rbf': SVC(kernel='rbf',  probability=True),
        'RandomForest': RandomForestClassifier(),
        #'kNN_DTW': KNeighborsTimeSeriesClassifier(metric='dtw'),
    }

    param_grids = {
        'LR': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']},
        'SVC_linear': {'C': [0.01, 0.1, 1, 10]},
        'SVC_rbf': {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'max_features': ['sqrt', 'log2']},
        #'kNN_DTW': {'n_neighbors': [1, 3, 5]}, 
    }

    best_params = {}
    results = {key: {'accuracy': [], 'f1': [], 'y_pred': [], 'y_test':[], 'll': []} for key in classifiers.keys()}

    for run in range(nb_iter):
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq=band,method_pca=method_pca,data_aug_method=data_aug_method,subj_included=subj_included,PC_use=PC_use, data_path=data_path, pol_cor=pol_cor)
        X_train, y_train = shuffle(X_train, y_train, random_state=run)
        if perm : 
            y_train = shuffle(y_train, random_state=0)
        for name, clf in classifiers.items():
            if run == 0:
                grid = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy')
                if name in ['kNN_DTW', 'TimeSeriesSVC']:
                    grid.fit(X_train[:, :, np.newaxis], y_train)
                else:
                    grid.fit(X_train, y_train)
                
                best_params[name] = grid.best_params_
                clf = clf.set_params(**best_params[name])
                results[name]['param'] = best_params[name]
            else:
                clf = clf.set_params(**best_params[name])
            
            if name in ['kNN_DTW', 'TimeSeriesSVC']:
                clf.fit(X_train[:, :, np.newaxis], y_train)
                y_pred = clf.predict(X_test[:, :, np.newaxis])
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
            if hasattr(clf, "predict_proba"):
                ll = log_loss(y_test, clf.predict_proba(X_test)[:,1])
            else:
                ll = np.nan
            
            results[name]['accuracy'].append(accuracy_score(y_test, y_pred))
            results[name]['f1'].append(f1_score(y_test, y_pred, average='macro'))
            results[name]['y_pred'].extend(y_pred)
            results[name]['y_test'].extend(y_test)
            results[name]['ll'].append(ll)

    #PlotCompareModels(band, method_pca, data_aug_method, results,pc_used=PC_use, save=save, perm=perm)
    if perm :
        l = '_permuted'
    else : 
        l=''

    # save result 
    if save :
        df_res = pd.DataFrame(results)
        if not os.path.exists(out_path + f'/{band}') :
            os.makedirs(out_path + f'/{band}')

        df_res.to_csv(out_path + f'/{band}/{method_pca}_{data_aug_method}{l}_{PC_use}_CompareModels.csv')
    else : 
        return df_res
    
def clean_df(x) : 
    return [float(i) for i in x.replace('[', '').replace(']', '').replace(' ', '').split(sep=',')]

def CompareModelsGGPlot(score, band, pc_use, decoding_folder=OUT_PATH + '/Decoding') : 
    methods = []
    means = []
    models = None

    color_map = {
        ('mean','mean'): 'tab:blue',
        ('mean','duplicat'): 'tab:orange',
        ('concat','mean'): 'tab:green',
        ('concat','duplicat'): 'tab:red'
    }

    for method_pca in ['mean', 'concat']:
        for data_aug_method in ['mean', 'duplicat']:
            tag = f"PCA={method_pca}, AUG={data_aug_method}"
            color = color_map[(method_pca, data_aug_method)]
            df = pd.read_csv(decoding_folder + f'/{band}/{method_pca}_{data_aug_method}_{pc_use}_CompareModels.csv').rename(columns={'Unnamed: 0':'scores'}).set_index('scores')
            df_perm = pd.read_csv(decoding_folder + f'/{band}/{method_pca}_{data_aug_method}_permuted_{pc_use}_CompareModels.csv').rename(columns={'Unnamed: 0':'scores'}).set_index('scores')

            df_perm.loc['y_test', :] =  df_perm.loc['y_test', :].apply(clean_df)
            df.loc['y_test', :] =  df.loc['y_test', :].apply(clean_df)
            df_perm.loc['y_pred', :] =  df_perm.loc['y_pred', :].apply(lambda x  : x.replace('np.int64(', '').replace(')', '')).apply(clean_df)
            df.loc['y_pred', :] =  df.loc['y_pred', :].apply(lambda x  : x.replace('np.int64(', '').replace(')', '')).apply(clean_df)

            if models is None:
                models = df.columns

            if score == 'f1' : 
                m= [f1_score(df.loc['y_pred', m], df.loc['y_test', m])  for m in models]
                m_p= [f1_score(df_perm.loc['y_pred', m], df_perm.loc['y_test', m])  for m in models]
            elif score == 'll':
                m = [np.mean(clean_df(df.loc['ll', m])) for m in models]
                m_p =[np.mean(clean_df(df_perm.loc['ll', m])) for m in models]
            else : 
                m= [accuracy_score(df.loc['y_pred', m], df.loc['y_test', m])  for m in models]
                m_p= [accuracy_score(df_perm.loc['y_pred', m], df_perm.loc['y_test', m])  for m in models]
                score = 'accuracy'

            methods.append((tag, color, 'normal'))
            means.append(m)
            methods.append((tag, color, 'perm'))
            means.append(m_p)

    x = np.arange(len(models))
    width = 0.08  # width of each bar
    offsets = np.linspace(-0.35, 0.35, len(methods))

    plt.figure(figsize=(16,6))
    plt.title(f"{score.capitalize()} across models -- {band} -- PC: {pc_use +1}", fontsize=20)

    for offset, (tag, color, mtype), m in zip(offsets, methods, means):
        
        label = tag + (" (Permuted)" if mtype == "perm" else "")
        hatch = "//" if mtype == "perm" else None

        plt.bar(
            x + offset,
            m,
            width,
            label=label,
            color=color,
            hatch=hatch,
            alpha=0.6
        )

    plt.ylim((0, 1))
  
    plt.xticks(x, models, rotation=45)
    plt.ylabel(score)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    plt.show()

################################### 'STATS' ###################################
def decodingTS(band, method_pca, data_aug_method,subj_included=[], iteration=100, PC_use=0, save=False, out_path=f'{OUT_PATH}/Decoding', iter_perm=50, data_path=OUT_PATH + '/Data', model_name = 'LR', crop_arg={'crop' :False, 't_id_min':0, 't_id_max':0 }) :
    truth = []

    # TO STORE
    Y_TEST = []
    Y_PRED = []
    Y_PRED_SH = []
    p= []
    p_sh  =[]
    weights_model = []
    weights_model_sh = []
    PCA_weights= []
    PCA_weights_sh= []

    # select model 
    clf, param_grid = select_model(model_name=model_name)

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
        subj_included = ExcludSubj(subj_included, data_path=data_path)

    for i in range(iteration):
        # PREP DATA
        TFRm_list = []
        Train_sample= []
        Test_sample =[]
        for subj in subj_included : 
            info_file = data_path + f'/{subj}_info.json'
            with open(info_file) as f:
                info = json.load(f)
                events_index = np.array([int(i) for i in info['event_id']])

            id_ev1 = np.where(events_index == 1)[0]
            id_ev2 = np.where(events_index == 2)[0]

            # Keep 1 id per condi for testing 
            id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
            id_ev1 = list(id_ev1)
            id_ev1.remove(id_test[0])
            id_ev1 = np.array(id_ev1)
            id_ev2 = list(id_ev2)
            id_ev2.remove(id_test[1])
            id_ev2 = np.array(id_ev2)
            
            # Compute mean over trials 
            if band == 'broadband' :
                TFRm = BbEvents(subj, test_id = id_test, events_index=events_index, data_path=data_path)
            else : 
                freq_id = FREQ_BAND.index(band)
                TFRm = TFRmEvents(subj, test_id = id_test, freq_id = freq_id, events_index=events_index, data_path=data_path)

            # Save for PCA computation at grp level
            if method_pca == 'concat' :
                TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = -1))
            if method_pca == 'mean' : 
                TFRm_list.append(np.mean(TFRm, axis = 0))

            del TFRm
            # Get the trilas data
            if band == 'broadband' :
                file = data_path + f'/{subj}_epochs.p'
                with open(file, "rb") as f:
                    TFRtr = pickle.load(f)  

                TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
                Test_sample.append(TFRtr[id_test,:, :])

            else :
                file = data_path + f'/{subj}_TFRtrials.p'
                with open(file, "rb") as f:
                    TFRtr = pickle.load(f)  

                # Augment the data
                TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_id, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
                Test_sample.append(TFRtr[id_test,:, freq_id, :])
            
            a = np.concat([TFRtr_augmented[None, :23, :, :], TFRtr_augmented[None, 23:, :, :]], axis = 0)
            perm_trials = np.random.permutation(a.shape[1])
            a[0, :, :, :] = a[0, perm_trials, :, :]
            perm_trials = np.random.permutation(a.shape[1])
            a[1, :, :, :] = a[1, perm_trials, :, :]
            TFRtr_augmented = np.concat([a[0, :, :, :], a[1, :, :, :]], axis = 0)

            Train_sample.append(TFRtr_augmented)
            truth.append(true_trials)
            
        concat_all = np.concatenate(TFRm_list, axis = 0) 
        Train_all = np.concatenate(Train_sample, axis=1)

        # create test sample and labels 
        y_train = [0]*23 + [1]*23
        Test_all = np.concatenate(Test_sample, axis =2)
        y_test = [0,1]
        Y_TEST.extend(y_test)

        # shuffle trials 
        perm_trials = np.random.permutation(Train_all.shape[0])
        Train_all = Train_all[perm_trials, :, :]
        y_train = [y_train[i] for i in perm_trials]
        
        del TFRtr
        del TFRm_list
        del Train_sample
        del Test_sample

        # NORMAL DECODING
        df_Componants, _ , means = ConcatPCA({'grp' : concat_all}, ch_id = False, nb_compo=PC_use+1, freq_band=[band], return_mean=True)
        weights = df_Componants['grp'].query("freq == @band").drop(columns = ['freq', 'compo']).values
        PCA_weights.append(weights[PC_use, :])

        # Shuffle electrods
        permutation_ch = np.random.permutation(concat_all.shape[0])
        Train_all = Train_all[:, permutation_ch, :] 
        weights = weights[:, permutation_ch]
        Test_all = Test_all[:, :, permutation_ch, :]

        mean_pca = means[band]
        m_train= mean_pca[None, :, None]
        m_test = mean_pca[None, :, None]

        if type(PC_use) == list :
            Train_transformed = np.zeros([Train_all.shape[0],len(PC_use), Train_all.shape[-1]])
            Test_transformed = np.zeros([Test_all.shape[0], len(PC_use),Test_all.shape[-1]])
            for pc in PC_use : 
                Train_transformed[:, pc, :] = weights[pc, :] @ (Train_all - m_train)
                Test_transformed[:, pc, :] = weights[pc, :] @ (Test_all[:,0,:] - m_test)
                
        else : 
            Train_transformed = weights[PC_use, :] @ (Train_all - m_train)
            Test_transformed = weights[PC_use, :] @ (Test_all[:,0,:] - m_test)

        # model BS?
        if crop_arg['crop'] : 
            Train_transformed = Train_transformed[:, crop_arg['t_id_min']:crop_arg['t_id_max']]
            Test_transformed = Test_transformed[:, crop_arg['t_id_min']:crop_arg['t_id_max']]
            l_crop = '_crp' + str(crop_arg['t_id_min']) + '_' + str(crop_arg['t_id_max']) + '_'
        else : 
            l_crop = ''

        if i == 0 :
            base_model = clone(clf)
            grid = GridSearchCV(base_model, param_grid, cv=5)
            grid.fit(Train_transformed, y_train)
            best_params = grid.best_params_
            del base_model

        model = clone(clf)
        model.set_params(**best_params)
        model.fit(Train_transformed, y_train)
        del Train_transformed

        # SAVE model not shuffle
        Y_PRED.extend(model.predict(Test_transformed))
        
        # save wiehgts and convergence metrics
        if model_name in 'LR': 
            p.append(log_loss(y_test, model.predict_proba(Test_transformed)[:,1]))
            weights_model.append(model.coef_)
        elif model_name == 'SVC_linear':
            weights_model.append(model.coef_)
            scores = model.decision_function(Test_transformed)
            p.append(hinge_loss([-1, 1], scores))
        elif model_name == 'RandomForest' : 
            weights_model.append(model.feature_importances_)
            p.append(model.oob_score_)
        elif model_name == 'SVC_rbf' :
            scores = model.decision_function(Test_transformed)
            p.append(hinge_loss([-1, 1], scores))
            weights_model.append(None)
        del model 
        del Test_transformed

        # SHUFFLED MODEL
        for j in range(iter_perm) : 
            # Shuffleing TFRm to compute PCA
            if len(concat_all.shape) == 3 :
                if method_pca == 'concat' : 
                    concat_ev1 = concat_all[:, freq_id, :int(concat_all.shape[-1]/2)]
                    concat_ev2 = concat_all[:, freq_id,  int(concat_all.shape[-1]/2):]
            
                    # shuffle the event to desconstruct the PC-concat orga ev1-- ev2
                    concat_ev12=np.concatenate([[concat_ev1], [concat_ev2]]) # got (2, channels, time) 
                    concat_ev12_shuffled = concat_ev12.copy()
                    np.random.shuffle(concat_ev12_shuffled)  # shuffle axis = 0
                    concat_all_sh = np.concat([concat_ev12_shuffled[i, :,:] for i in range(2)], axis = -1)
                    
                else : 
                    concat_all_sh = concat_all[:, freq_id, :]
            else:
                if method_pca == 'concat' :
                    concat_ev1 = concat_all[:, :int(concat_all.shape[-1]/2)]
                    concat_ev2 = concat_all[:, int(concat_all.shape[-1]/2):]
                    # shuffle the event to desconstruct the PC-concat orga ev1-- ev2
                    concat_ev12=np.concatenate([[concat_ev1], [concat_ev2]]) # got (2, channels, time) 
                    concat_ev12_shuffled = concat_ev12.copy()
                    np.random.shuffle(concat_ev12_shuffled)  # shuffle axis = 0
                    concat_all_sh = np.concat([concat_ev12_shuffled[i, :,:] for i in range(2)], axis = -1)
                else : 
                    concat_all_sh = concat_all

            df_Componants_sh, _, means = ConcatPCA({'grp' : concat_all_sh}, ch_id = False, nb_compo=PC_use+1, freq_band=[band], return_mean=True)
            del concat_all_sh

            weights_sh = df_Componants_sh['grp'].query("freq == @band").drop(columns = ['freq', 'compo']).values
            PCA_weights_sh.append(weights[PC_use, :])
            mean_pca = means[band]
            m_train= mean_pca[None, :, None]
            m_test = mean_pca[None, :, None]

            # Applied on train 
            if type(PC_use) == list : 
                Train_transformed_sh = np.zeros([Train_all.shape[0],len(PC_use), Train_all.shape[-1]])
                Test_transformed_sh = np.zeros([Test_all.shape[0], len(PC_use),Test_all.shape[-1]])
                for pc in PC_use : 
                    Train_transformed_sh[:, pc, :] = weights_sh[pc, :] @ (Train_all -m_train)
                    Test_transformed_sh[:, pc, :] = weights_sh[pc, :] @ (Test_all[:,0,:] -m_test)
                    
            else : 
                Train_transformed_sh = weights_sh[PC_use, :] @ Train_all
                Test_transformed_sh = weights_sh[PC_use, :] @ Test_all[:,0,:]
            
            if crop_arg['crop'] : 
                Train_transformed_sh = Train_transformed_sh[:, crop_arg['t_id_min']:crop_arg['t_id_max']]
                Test_transformed_sh = Test_transformed_sh[:, crop_arg['t_id_min']:crop_arg['t_id_max']]
                l_crop = '_crp' + str(crop_arg['t_id_min']) + '_' + str(crop_arg['t_id_max']) + '_'
            else : 
                l_crop = ''

            # Shuffle the labels
            y_train_sh = shuffle(y_train)         
            # Applied the model
            model_sh = clone(clf)
            model_sh.set_params(**best_params)
            model_sh.fit(Train_transformed_sh, y_train_sh)
            del Train_transformed_sh
            # SAVE model shuffle
            Y_PRED_SH.extend(model_sh.predict(Test_transformed_sh))

            if model_name in 'LR': 
                p_sh.append(log_loss(y_test, model_sh.predict_proba(Test_transformed_sh)[:,1]))
                weights_model_sh.append(model_sh.coef_)
            elif model_name == 'SVC_linear':
                weights_model_sh.append(model_sh.coef_)
                scores = model_sh.decision_function(Test_transformed_sh)
                p_sh.append(hinge_loss([-1, 1], scores))
            elif model_name == 'RandomForest' : 
                weights_model_sh.append(model_sh.feature_importances_)
                p_sh.append(model_sh.oob_score_)
            elif model_name == 'SVC_rbf' :
                scores = model_sh.decision_function(Test_transformed_sh)
                p_sh.append(hinge_loss([-1, 1], scores))
                weights_model_sh.append(None)
            
            del Test_transformed_sh
            del model_sh

    sumsum= pd.DataFrame()
    sumsum['band'] =[band]
    sumsum['method_pca'] = [method_pca]
    sumsum['data_aug_method'] = [data_aug_method]
    sumsum['iter'] = [iteration]
    sumsum['iter_perm']=[iter_perm]
    sumsum['y_pred'] = [Y_PRED]
    sumsum['y_pred_sh'] = [Y_PRED_SH]
    sumsum['y_test'] = [Y_TEST]
    sumsum['entropy'] = [p]
    sumsum['entropy_sh'] = [p_sh]
    sumsum['weight'] = [weights_model]
    sumsum['weight_sh'] = [weights_model_sh]
    sumsum['pca_weight'] = [PCA_weights]
    sumsum['pca_weight_sh'] = [PCA_weights_sh]
    sumsum['best_param'] = [best_params]

    if save : 
        if not os.path.isdir(out_path + f'/{band}/fullmodel') : 
            os.makedirs(out_path + f'/{band}/fullmodel')
            
        sumsum.to_csv(out_path + f'/{band}/fullmodel/{method_pca}_{data_aug_method}_{PC_use}_{model_name}full{l_crop}.csv')

    else : 
        return sumsum
    
def select_model(model_name) : 
    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'SVC_linear': SVC(kernel='linear'),
        'SVC_rbf': SVC(kernel='rbf',  probability=True),
        'RandomForest': RandomForestClassifier(oob_score=True),
    }

    param_grids = {
        'LR': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'liblinear']},
        'SVC_linear': {'C': [0.01, 0.1, 1, 10]},
        'SVC_rbf': {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'max_features': ['sqrt', 'log2']},
    }

    return classifiers[model_name], param_grids[model_name]

def cleandfdecodingTS(df_final, shape_time) : 
    return_dict ={}

    for f in ['acc', 'acc_sh', 'pca_weight', 'pca_weight_sh', 'weight_mean', 'weight_std', 'weight_sh_mean','weight_sh_std', 'entropy', 'entropy_sh_mean', 'entropy_sh_std']: 
        return_dict[f] =[]

    y_pred = np.array([int(x.replace('[', '').replace('np.int64(', '').replace(')', '').replace(']', '')) for x in df_final.y_pred.values[0].split(', ')])
    y_pred_sh = np.array([int(x.replace('[', '').replace('np.int64(', '').replace(')', '').replace(']', '')) for x in df_final.y_pred_sh.values[0].split(', ')]) # 100 * 100 iteration
    y_pred_sh = y_pred_sh.reshape(int(df_final.loc[0, 'iter'])*2, int(df_final.loc[0, 'iter_perm']))
    y_test = np.array([int(x.replace('[', '').replace(']', '')) for x in df_final.y_test.values[0].split(', ')])
    entropy= np.array([float(x.replace('[', '').replace('np.float64(', '').replace(')', '').replace(']', '')) for x in df_final.entropy.values[0].split(', ')])
    entropy_sh= np.array([float(x.replace('[', '').replace('np.float64(', '').replace(')', '').replace(']', '')) for x in df_final.entropy_sh.values[0].split(', ')])
    entropy_sh = entropy_sh.reshape(int(df_final.loc[0, 'iter']), int(df_final.loc[0, 'iter_perm']))  
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.weight.values[0]))
    weight = np.fromstring(numbers_only, sep=' ').reshape(-1, shape_time)
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.weight_sh.values[0]))
    try :     
        weight_sh = np.fromstring(numbers_only, sep=' ').reshape(int(df_final.loc[0, 'iter']), int(df_final.loc[0, 'iter_perm']), shape_time) # TO check
        return_dict['weight_sh_mean'] = weight_sh.reshape(-1, shape_time).mean(0)
        return_dict['weight_sh_std'] = weight_sh.reshape(-1, shape_time).std(0)
        return_dict['weight_sh'] = weight_sh.reshape(-1, shape_time)
    except : 
         return_dict['weight_sh'] = None
    
    return_dict['weight'] = weight
    return_dict['weight_mean'] = weight.mean(0)
    return_dict['weight_std'] = weight.std(0)
    return_dict['y_pred'] = y_pred
    return_dict['y_pred_sh'] = y_pred_sh
    return_dict['y_test'] = y_test

    # compute accuracy 
    acc_along_perm=[]  
    for i in range(int(df_final.loc[0, 'iter_perm'])) : 
        acc_along_perm.append(accuracy_score(y_pred_sh[:, i], y_test))
    
    return_dict['acc_sh'] = np.mean(acc_along_perm) # acc global with the stability of perms
    return_dict['acc'] = accuracy_score(y_pred, y_test) # acc global

    # weights pca
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.pca_weight.values[0]))
    return_dict['pca_weight'] = np.fromstring(numbers_only, sep=' ')
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.pca_weight_sh.values[0]))
    return_dict['pca_weight_sh'] = np.fromstring(numbers_only, sep=' ')
    
    # entropy 
    return_dict['entropy'] = entropy
    return_dict['entropy_sh'] = entropy_sh
    return_dict['entropy_sh_mean'] = np.array(entropy_sh).mean(axis=1)
    return_dict['entropy_sh_std']  = np.array(entropy_sh).std(axis=1)

    return return_dict
 