
import os
from utils import OUT_PATH, PROJECT_PATH
from utils import GetInfo,  ExcludSubj

epoch_path = OUT_PATH + '/Data_shortWOBS'

# Get subject
subj_included = [file.replace('_epochs.p', '') for file in os.listdir(epoch_path) if file[-len('epochs.p'):] == 'epochs.p']
subj_included = ExcludSubj(subj_included,  data_path=epoch_path)
print('Number of subject is ', len(subj_included))

# Get info 
GetInfo(subj_included, project_path = PROJECT_PATH, data_path = OUT_PATH + '/Data_shortWOBS', save=True)
