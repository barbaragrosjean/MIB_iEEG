# Function 1. load_and_filter() 
# Get raw iEEG data
# Follow the preprocessing in preproc.py but do not downsample and do not compute the power
# filtered between 80 and 120 Hz using a second-order Butterworth filter.
# A Hilbert transform was then applied to the filtered signal to extract its instantaneous amplitude. 
# OUTPUT: matrix (condition, trials, time) for each subject in outs/ripples/enveloppes

# Function 2. identify_ripples()
# Take the envolopes and compute the ripples for each subject 
# Ripples were identified as events with a maximum amplitude three s.d. above the mean, and with a duration 
# of at least 25 ms. Ripples that were detected within 15 ms of each other were merged. 
# OUTPUT: should be a dataframe with the subject, electrodes, area of interest, timing, 
# duration and all the ripples metrics for each ripples detected

# Function 3. test_adjacent()
# For patients who had more than one anterior hippocampal electrode we examined whether the same ripple
# was picked up by the different electrodes (within 20 ms).
# OUTPUT : percent of simulteneous ripples in paired electrodes per subject (save it)

# Function 4. subj_ripples_trials()
# OUTPUT a figure of a 2d histogram trial vs electrodes, of the ripples count over time for each trials and electrodes
# I want that in the annotation ylabel we see the area in which the electrodes is implented. 
# For each subject.

# Function 5. subj_ripples_in_time()
# OUTPUT a figure of a 2d histogram time vs electrodes, of the ripples count over trials I want that in the annotation ylabel 
# we see the area in which the electrodes is implented. For each subject. you should define time bins to count over the trials.

# Function 6. group_ripples_in_time()
# Same than above but group all participant, you should have for each subject the average count over trials:
# (condition, electrodes, time_bins ). you should group the participant and divid by regions of interest. I want one plot for each RIOs
# then an figure for the average taking the average count per region of interest so a 2d plot of ROIs x time at group level. 
