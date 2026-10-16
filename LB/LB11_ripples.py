import os
import pickle

from src.setting import ExcludSubj
from src.config import OUT_PATH, PROJECT_PATH, FREQS, BWIDTH, EVENT_ID
from src.preproc import preproc

subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']
subj_included = ExcludSubj(subj_included=subj_included, data_path=OUT_PATH + '/Data_longWOBS')

failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']

for s in failled : 
    if s in subj_included :
        subj_included.remove(s)



def load_and_filter(subj, out_path=OUT_PATH + '/ripples/enveloppes'):
    if not os.path.exists(out_path) : 
        os.makedirs(out_path)

    # Get raw iEEG data and preproc without downsampling
    epoch = preproc(subj, sfreq = 600,new_sfreq = 'none', freqs = FREQS, 
            bwidth = BWIDTH, event_id = EVENT_ID, project_path = PROJECT_PATH, 
            out_path = out_path, trials = True, save_epoch=False, compute_TFR=False)  

    # filtered between 80 and 120 Hz using a second-order Butterworth filter.


    # A Hilbert transform was then applied to the filtered signal to extract its instantaneous amplitude. 

    # SAVE OUTPUT: matrix (condition, trials, time) of enveloppe


def identify_ripples(subj, env_path=OUT_PATH + '/ripples/enveloppes', out_path=OUT_PATH + '/ripples/counts') : 
    # Take the envolopes and compute the ripples for each subject 
    # Ripples were identified as events with a maximum amplitude three s.d. above the mean, and with a duration 
    # of at least 25 ms. Ripples that were detected within 15 ms of each other were merged. 
    # OUTPUT: should be a dataframe with the subject, electrodes, area of interest, timing,  stored in the outpath
    # duration and all the ripples metrics for each ripples detected


def test_adjacent(subj, env_path=OUT_PATH + '/ripples/counts', out_path=OUT_PATH + '/ripples/adjacent') :

    # For patients who had more than one anterior hippocampal electrode we examined whether the same ripple
    # was picked up by the different electrodes (within 20 ms).
    # OUTPUT : percent of simulteneous ripples in paired electrodes per subject (save it in )


def subj_ripples_trials(subj, ):
    # OUTPUT a figure of a 2d histogram trial vs electrodes, of the ripples count over time for each trials and electrodes
    # I want that in the annotation ylabel we see the area in which the electrodes is implented. 
    # For each subject.
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be subj + _trials

def subj_ripples_in_time(subj, ):
    # OUTPUT a figure of a 2d histogram time vs electrodes, of the ripples count over trials I want that in the annotation ylabel 
    # we see the area in which the electrodes is implented. For each subject. you should define time bins to count over the trials.
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be subj + _time


def group_ripples_in_time(subj_included):
    # Same than above but group all participant, you should have for each subject the average count over trials:
    # (condition, electrodes, time_bins ). you should group the participant and divid by regions of interest. I want one plot for each RIOs
    # then an figure for the average taking the average count per region of interest so a 2d plot of ROIs x time at group level. 
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be group 


def ripples_pipline(subject_included) :
    for subj in subject_included:
        load_and_filter(subj)
        identify_ripples(subj)
        test_adjacent(subj)

    

