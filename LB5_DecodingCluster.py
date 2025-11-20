
import os
from utils import OUT_PATH, FREQ_BAND
from utils import PermLR_distrib, PermLR_null, LR, TemporalLR, TemporalGeneralization, ExcludSubj, TemporalLRRaw, TemporalGeneralizationRaw, CompareClassifier
import warnings

if __name__ == "__main__":
    iteration =100
    iter_perm = 50
    perm =True
    pc_use = 0
    warnings.filterwarnings("ignore", category=FutureWarning)

    band_to_try  = FREQ_BAND + ['broadband']
    method_pca_to_try = ['mean', 'concat']
    data_aug_method_to_try = ['mean', 'duplicat']

    subj_included_ = [file.replace('_TFRtrials.p', '') for file in os.listdir(OUT_PATH + '/Data') if file[-len('_TFRtrials.p'):] == '_TFRtrials.p']
    subj_included = ExcludSubj(subj_included_)
    
    for band in ['high_gamma'] : # FREQ_BAND + ['broadband'] :
        for data_aug_method in ['duplicat', 'mean'] :
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
            
            for method_pca in ['concat', 'mean'] :
                #####################
                # Compute decoding
                #####################

                LR(band=band, 
                    perm=perm,
                    method_pca=method_pca, 
                    data_aug_method=data_aug_method, 
                    subj_included=subj_included, 
                    PC_use=pc_use, 
                    save=True, 
                    iter_perm=iter_perm)
                    
                #TemporalLR(band=band, 
                #            method_pca=method_pca, 
                #            data_aug_method=data_aug_method, 
                #            subj_included=subj_included, 
                #            PC_use=pc_use, 
                #            save=True)
                    
                #TemporalGeneralization(band=band, 
                #                        method_pca=method_pca, 
                #                        data_aug_method=data_aug_method, 
                #                        subj_included=subj_included, 
                #                        PC_use = pc_use,
                #                        save=True)
                
                #CompareClassifier(band=band,
                #                method_pca=method_pca, 
                #                data_aug_method=data_aug_method, 
                #                subj_included=subj_included,
                #                PC_use=pc_use, 
                #                perm = True,
                #                save=True)
                
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
                PermLR_distrib(band=band, 
                               method_pca=method_pca,
                                data_aug_method=data_aug_method,
                                subj_included = subj_included, 
                                iteration=iteration, 
                                PC_use=pc_use, 
                                save=True, 
                                iter_perm=iter_perm)
                
                PermLR_null(band=band, 
                            method_pca=method_pca, 
                            data_aug_method=data_aug_method,
                            subj_included=subj_included,
                            iteration=iteration, 
                            PC_use=pc_use, 
                            save=True,
                            iter_perm=iter_perm, 
                            accuracy = True, 
                            entropy=True)

