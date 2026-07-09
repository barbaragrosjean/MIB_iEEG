import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
plt.style.use('petroff10')

PROJECT_PATH = '../MINDLAB2021_MEG-TempSeqAges/scratch/learning_bach_iEEG'
OUT_PATH = 'outs'    
EVENT_ID = {'old/correct': 1,
 'new/correct': 2,
 'old/incorrect': 101,
 'new/incorrect': 102,
 'old/null': 201,
 'new/null': 202}

FREQS = [0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,21,24,27,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150,160,180]
BWIDTH = np.array([0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,20])

FREQ_BAND_DICT = {
    'Delta': [0.5, 1, 2, 3, 4],
    'Theta': [4, 5, 6, 7, 8],
    'Alpha': [8, 9, 10, 11, 12, 13],
    'Low_Beta': [13, 15, 17, 19],
    'High_Beta': [21, 24, 27, 30],
    'Low_Gamma': [30, 35, 40, 45, 50, 55],
    'High_Gamma': [55, 60, 70, 80, 90, 100]
}

FREQ_BAND = ['delta', 'theta','alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']

REGION = {'parietal': ['IPS','IP','SP','SPL','AG','SMG','TPJ'],
        #'premotor': ['SFG','SFS','MFG','FEF','SMA'],
        #'DLPFC': ['MFG','FEF','SFS','IFS'],
        'premotor' : ['SFG','FEF','SMA'],
        'DLPFC': ['MFG','SFS','IFS'],
        'M1': ['preCG','M1','PreCG'],
        'S1': ['postCG','PostCG'],
        'INS': ['INS'],
        'VLPFC': ['IFG','FOP','IFS'],
        #'MTL': ['EC','HPC','MEC','PRH','PHG','PHC','LEC'],
        'EC': ['EC','MEC','LEC'],
        'PRH' : ['PRH'],
        'A1': ['A1'],
        'MTG': ['MTG'],
        'AMY': ['AMY'],
        'PCC': ['PCC'],
        'ACC': ['ACC','MCC'],
        'HPC': ['HPC'],
        'PHC': ['PHC','PHG'],
        'STG': ['STG'],
        'STS': ['STS'],
        'TP': ['TP'],
        'OFC': ['OFC'],
        'VS': ['LG','FUG','ITG','ITS'], #ventral stream
        'THAL': ['THAL']}

COL_REG = {
    # --- Memory system (distinct blues)
    'HPC':  '#0b2e8a',   # deep navy
    'PHC':  '#2457c5',   # strong blue
    'EC':  '#6fa3ff',   # light blue
    'PCC':  '#3b4f9f',   # blue–indigo
    'PRH' : '#528EA4',

    # --- Limbic / affective / reward (distinct purples)
    'AMY':  '#5b1a8e',   # deep violet
    'VS':   '#8e44ad',   # saturated purple
    'OFC':  '#c39bd3',   # light lavender

    # --- Prefrontal / executive control (clearly separated reds/oranges)
    'DLPFC':"#7c1f16",   # dark red
    'ACC':  "#aa3124",   # red
    'VLPFC':"#b46433",   # orange

    # --- Sensory–motor (clearly separated greens)
    'M1':       '#145a32',   # dark green
    'premotor': '#229954',   # medium green
    'S1':       '#7dcea0',   # light green

    # --- Auditory / temporal association (strongly separated pink–reds)
    'A1':  "#e00101",   # dark rose
    'STG': "#ae1a3d",   # crimson pink
    'STS': '#e04b78',   # saturated pink
    'MTG': "#fc92b2",   # light pink
    'TP':  "#fbb1b1",   # very pale rose

    # --- Insula (salience bridge: blue–cyan, isolated)
    'INS': '#f1c40f',

    # --- Parietal association (yellow–green, isolated)
    'parietal': '#a3cb38',

    # --- Thalamic relay (olive–brown, isolated)
    'THAL': '#7f6d1f',

    # --- Non-brain / baseline
    'N': '#b3b3b3'
}

TASK = 'MusicMemory'

# Design
events = ['old/correct', 'new/correct']
event_idx = {0: 'Old', 1:'New'}
color_event = {0: 'b', 1:'r'}

col_pc = {0:"#0072B2", 1:"#D55E00", 2:"#009E73", 4:"#009E73"}
cmap_pc0 = LinearSegmentedColormap.from_list("pc0_blue",["#FFFFFF", "#77ABC8"])
cmap_pc1 = LinearSegmentedColormap.from_list("pc1_orange",["#FFFFFF", "#E1AB82"])
cmap_pc2 = LinearSegmentedColormap.from_list("pc2_green",["#FFFFFF", "#74C4AF"])
cmap_pcs = [cmap_pc0, cmap_pc1, cmap_pc2]


