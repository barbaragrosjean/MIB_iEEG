import os
from src.config import OUT_PATH, PROJECT_PATH
from src.preproc import preproc
from src.setting import ExcludSubj
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    subj_included = [file.replace('_epochs.p', '') for file in os.listdir(OUT_PATH + '/Data_longWOBS') if file[-len('epochs.p'):] == 'epochs.p']
    subj_included = ExcludSubj(subj_included=subj_included, data_path=OUT_PATH + '/Data_longWOBS')
    
    #subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']

    failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']
    
    for s in failled : 
        if s in subj_included :
            subj_included.remove(s)
    path_to_save = 'iEEGvsMEG/ieeg_shortWOBS_fs400' #OUT_PATH + 

    if not os.path.exists(path_to_save) :
        os.makedirs(path_to_save)

    for subj in subj_included :  
        if not os.path.exists(path_to_save + f'/{subj}_TFRtrials.p'):
            print(subj)
            preproc(subj, trials=True, save_epoch=True, compute_TFR=False, out_path=path_to_save,new_sfreq = 400 )
    