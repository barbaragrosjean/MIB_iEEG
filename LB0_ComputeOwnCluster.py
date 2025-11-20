import os
from utils import PROJECT_PATH, OUT_PATH
from utils import preproc
import warnings



if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']
    failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']
    path_to_save = OUT_PATH + '/Data'

    for subj in  ['BJH029', 'DA037', 'BJH058'] : 
        if subj not in failled : #and not os.path.exists(path_to_save + f'/{subj}_TFRtrials.p'):
            print(subj)
            preproc(subj, trials=True, save_epoch=True, compute_TFR=True)
     