
import os
from src.config import  OUT_PATH, FREQ_BAND
from src.decoding import decodingTS, LR, TemporalLR, TemporalGeneralization, TemporalLRRaw, TemporalGeneralizationRaw, CompareClassifier
from src.setting import ExcludSubj

import json
import warnings
import argparse
import numpy as np

def main_old(band, pc_use):
    iteration =100
    iter_perm = 5
    perm =False
    data_path = OUT_PATH + '/Data_LongWOBS'

    warnings.filterwarnings("ignore", category=FutureWarning)

    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    
    for data_aug_method in ['mean'] : #, 'duplicat'] :
        #####################
        # Decoding on Raw
        #####################
        #TemporalLRRaw(band=band, 
        #              data_aug_method= data_aug_method, 
        #              subj_included=subj_included, 
        #              save=True) 

        #TemporalGeneralizationRaw(band=band, 
        #                          data_aug_method=data_aug_method,
        #                          subj_included=subj_included, 
        #                          save=True)
        
        for method_pca in ['mean'] : #, 'concat'] :

            #####################
            # Compute decoding
            #####################

            LR(band=band,
                perm=perm,
                iteration=iteration,
                method_pca=method_pca, 
                data_aug_method=data_aug_method, 
                subj_included=subj_included, 
                PC_use=pc_use, 
                save=True, 
                iter_perm=iter_perm, 
                data_path=data_path)
                
            TemporalLR(band=band, 
                        method_pca=method_pca, 
                        data_aug_method=data_aug_method, 
                        subj_included=subj_included, 
                        PC_use=pc_use, 
                        save=True, 
                        data_path=data_path)
                
            TemporalGeneralization(band=band, 
                                    method_pca=method_pca, 
                                    data_aug_method=data_aug_method, 
                                    subj_included=subj_included, 
                                    PC_use = pc_use,
                                    save=True,
                                    data_path=data_path)
            
            #CompareClassifier(band=band,
            #                method_pca=method_pca, 
            #                data_aug_method=data_aug_method, 
            #                subj_included=subj_included,
            #                PC_use=pc_use, 
            #                nb_iter = iteration,
            #                perm = True,
            #                save=True, 
            #                data_path=data_path)
            
            #CompareClassifier(band=band,
            #                method_pca=method_pca, 
            #                data_aug_method=data_aug_method, 
            #                subj_included=subj_included,
            #                PC_use=pc_use, 
            #                nb_iter=iteration,
            #                perm = False,
            #                save=True, 
            #                data_path=data_path)
            
            
            #####################
            # Combine 2 componants
            #####################

            TemporalLRRaw(band=band, 
                          data_aug_method=data_aug_method,
                          subj_included=subj_included, 
                          iteration=iteration, 
                          PC_use=False, 
                          method_pca=method_pca, 
                          save=True)

            #TemporalGeneralizationRaw(band=band, 
            #                          data_aug_method=data_aug_method, 
            #                          subj_included=subj_included, 
            #                          PC_use=[0, 1], 
            #                          method_pca=method_pca, 
            #                          save=True, 
            #                          undersampling=False)
            
def mainTS(band, pc_use, model, method_pca, data_aug_method, bs_decoding=False):
    tfr_path = OUT_PATH + '/Data_longWOBS_mf70-160'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=tfr_path)
    subj_included_restricted = {16: ['BJH072', 'LL36', 'BJH069', 'SLCH020', 'BJH045', 'LL14', 'BJH050',
       'BJH027', 'OS70', 'BJH052', 'OS61', 'BJH041', 'BJH046', 'BJH056',
       'LL31', 'BJH039', 'BJH042', 'BJH026', 'LL08', 'SLCH018', 'BJH029',
       'DA037', 'BJH058', 'SLCH024', 'BJH049']}
    subj_included = subj_included_restricted[16]

    # get time 
    if bs_decoding :
        path = tfr_path + f'/{subj_included[0]}_info.json'
        with open(path) as json_data:
            d = json.load(json_data)
            if band == 'broadband':
                time = d['time_epoch']
            else : 
                time = d['time_tfr']
        
        #crop_arg = {'crop' : True, 't_id_min':time.index(np.array(time)[np.array(time) > -0.5][0]), 't_id_max' : len(time)}
        crop_arg = {'crop' : True, 't_id_min':0, 't_id_max' :time.index(np.array(time)[np.array(time) > -0.5][0]) }

    else : 
        crop_arg = {'crop' : False, 't_id_min':None, 't_id_max' : None}

    iteration = 100
    iter_perm = 100
    nb_trials = 16 #24
    decodingTS(band, 
            method_pca, 
            data_aug_method,
            subj_included, 
            iteration=iteration, 
            PC_use=pc_use, 
            save=True, 
            out_path=f'{OUT_PATH}/Decoding_shuffled_trials_mf70-160_res', 
            iter_perm=iter_perm, 
            data_path=tfr_path, 
            model_name = model, 
            crop_arg=crop_arg, 
            n_tr=nb_trials)
    
def mainCompareClf(band, pc_use):
    data_path = OUT_PATH + '/Data_longWOBS'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    iteration = 100

    for perm in [True, False] :
        for data_aug_method in ['mean', 'duplicat'] : 
            for method_pca in ['mean', 'concat'] : 
                CompareClassifier(band=band,
                                method_pca=method_pca, 
                                data_aug_method=data_aug_method, 
                                subj_included=subj_included,
                                PC_use=pc_use, 
                                nb_iter = iteration,
                                perm = perm,
                                save=True, 
                                data_path=data_path)

def TSwTemporalMasking(band, pc_use, model, method_pca, data_aug_method, bs_decoding=False):
    tfr_path = OUT_PATH + '/Data_longWOBS'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=tfr_path)

    iteration = 100
    iter_perm = 50

    # get time 
    path = tfr_path + f'/{subj_included[0]}_info.json'
    with open(path) as json_data:
        d = json.load(json_data)
        if band == 'broadband':
            time = d['time_epoch']
        else : 
            time = d['time_tfr']

    # slice timming into 10 slices 
    id_time = np.linspace(0, len(time), 10, dtype = int)
    id_time = id_time[7:]
    for i in range(len(id_time)-1) : 
        crop_arg = {'crop' : True, 't_id_min':id_time[i], 't_id_max' : id_time[i+1]}#id_time[i]
        decodingTS(band, 
                method_pca, 
                data_aug_method,
                subj_included, 
                iteration=iteration, 
                PC_use=pc_use, 
                save=True, 
                out_path=f'{OUT_PATH}/Decoding_shuffled_trials', 
                iter_perm=iter_perm, 
                data_path=tfr_path, 
                model_name = model, 
                crop_arg=crop_arg)
        
def mainTG(band, method_pca, data_aug_method):
    tfr_path = OUT_PATH + '/Data_longWOBS_full'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=tfr_path)

    TemporalGeneralizationRaw(band, 
                              data_aug_method, 
                              subj_included, 
                            save=True,
                            PC_use=[0, 1],
                            method_pca=method_pca, 
                            undersampling=True, 
                            data_path =tfr_path)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specific frequency band")
    parser.add_argument("--band", type=str, choices=FREQ_BAND + ['broadband'], required=False,
                        help="Frequency band to process.")
    
    parser.add_argument("--pc_use", type=int, choices=[0, 1, 2, 3, 4, 5], required=False,
                        help="PC to use to process.")
    
    parser.add_argument("--model", type=str, choices=["LR", "SVC_linear", "SVC_rbf", "RandomForest"], required=False,
                        help="Model to run.")

    parser.add_argument("--method_pca", type=str, choices=["concat", "mean"], required=False,
                        help="PCA method to select.")
    
    parser.add_argument("--method_data_aug", type=str, choices=["mean", "duplicat"], required=False,
                        help="Method for data augmentation.")
    
    args = parser.parse_args()

    #TSwTemporalMasking(args.band, args.pc_use, 'SVC_rbf', 'mean', args.method_data_aug, bs_decoding=False)

    for pc_use in [0, 1, 2] : 
        print('Pc:', pc_use)
        for model in ['RandomForest'] : 
            print('Model:', model)
            mainTS('high_gamma', pc_use, model, 'concat', None, bs_decoding=False)


    # TemporalLRRaw(band=args.band, data_aug_method=args.method_data_aug, 
    #               data_path = OUT_PATH + '/Data_longWOBS',iteration=100, 
    #               PC_use=False, method_pca=False, 
    #               save=True, out_path=f'{OUT_PATH}/Decoding_shuffled_trials')
    
    #mainTS(args.band, args.pc_use, args.model, args.method_pca, args.method_data_aug, bs_decoding=True)
    #mainCompareClf(args.band, args.pc_use)
    #mainTG(args.band, args.method_pca, args.method_data_aug)
