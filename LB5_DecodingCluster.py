
import os
from utils import OUT_PATH, FREQ_BAND
from utils import PermLR_distrib, PermLR_null, LR, TemporalLR, TemporalGeneralization, ExcludSubj, TemporalLRRaw, TemporalGeneralizationRaw, CompareClassifier, PermLR_Final
import warnings
import argparse

def main(band, pc_use):
    iteration =100
    iter_perm = 100
    perm =False
    data_path = OUT_PATH + '/Data_shortWOBS'

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
        
        for method_pca in ['concat'] : #, 'mean'] :

            #####################
            # Compute decoding
            #####################

            #LR(band=band,
            #    perm=perm,
            #    iteration=iteration,
            #    method_pca=method_pca, 
            #    data_aug_method=data_aug_method, 
            #    subj_included=subj_included, 
            #    PC_use=pc_use, 
            #    save=True, 
            #    iter_perm=iter_perm, 
            #    data_path=data_path)
                
            #TemporalLR(band=band, 
            #            method_pca=method_pca, 
            #            data_aug_method=data_aug_method, 
            #            subj_included=subj_included, 
            #            PC_use=pc_use, 
            #            save=True, 
            #            data_path=data_path)
                
            #TemporalGeneralization(band=band, 
            #                        method_pca=method_pca, 
            #                        data_aug_method=data_aug_method, 
            #                        subj_included=subj_included, 
            #                        PC_use = pc_use,
            #                        save=True,
            #                        data_path=data_path)
            
            #CompareClassifier(band=band,
            #                method_pca=method_pca, 
            #                data_aug_method=data_aug_method, 
            #                subj_included=subj_included,
            #                PC_use=pc_use, 
            #                perm = False,
            #                save=True, 
            #                data_path=data_path)
            
            #CompareClassifier(band=band,
            #                method_pca=method_pca, 
            #                data_aug_method=data_aug_method, 
            #                subj_included=subj_included,
            #                PC_use=pc_use, 
            #                perm = True,
            #                save=True, 
            #                data_path=data_path)
            
            
            #####################
            # Combine 2 componants
            #####################

            #TemporalLRRaw(band=band, 
            #              data_aug_method=data_aug_method,
            #              subj_included=subj_included, 
            #              iteration=iteration, 
            #              PC_use=[0, 1], 
            #              method_pca=method_pca, 
            #              save=True)

            #TemporalGeneralizationRaw(band=band, 
            #                          data_aug_method=data_aug_method, 
            #                          subj_included=subj_included, 
            #                          PC_use=[0, 1], 
            #                          method_pca=method_pca, 
            #                          save=True, 
            #                          undersampling=False)
            
            #####################
            # Permutation scores
            #####################
            
            PermLR_Final(band=band, 
                        method_pca=method_pca, 
                        data_aug_method=data_aug_method,
                        subj_included=subj_included,
                        iteration=iteration, 
                        PC_use=pc_use, 
                        save=True, 
                        iter_perm=iter_perm, 
                        data_path=data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specific frequency band")
    parser.add_argument("--band", type=str, choices=FREQ_BAND + ['broadband'], required=True,
                        help="Frequency band to process.")
    parser.add_argument("--pc_use", type=int, choices=[0, 1, 2], required=True,
                        help="PC to use to process.")
    
    args = parser.parse_args()
    main(args.band, args.pc_use)