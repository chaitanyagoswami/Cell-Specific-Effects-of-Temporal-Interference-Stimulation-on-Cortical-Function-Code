import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
from elec_field import sparse_place_human
from pulse_train import PulseTrain_Sinusoid, PulseTrain_TI
import sys
import math

SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))
SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/Non-invasive')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#### Defining Electric Field Simulator
##################################################################################
##################################################################################

start_time = time.time()
print("Loading Electric Field Simulator...")
overall_radius = 9.2 ## Radius of the sphere representing the whole head
elec_field_id = int(sys.argv[1])
if elec_field_id == 0:
    #### Defing the SAVING Directory for C3-C4 Location
    SAVE_PATH = os.path.join(SAVE_PATH, 'C3-C4')
 
elif elec_field_id == 1:
    #### Defing the SAVING Directory for C3-C4 Location
    SAVE_PATH = os.path.join(SAVE_PATH, 'C3-Cz')
elif elec_field_id == 2:
    #### Defing the SAVING Directory for C3-C4 Location
    SAVE_PATH = os.path.join(SAVE_PATH, 'HD-TDCS')
else:
    raise Exception('Wrong Id Supplied for Electrode Configuration. Valid Options: 0->C3C4; 1->C3Cz; 2->HD-TDCS')

amp_level = np.load(os.path.join(SAVE_PATH,"Amplitude.npy"))
diff_TI_lst, diff_sin_lst = [], []
error_TI_lst, error_sin_lst = [], []
pyr_TI_lst, pyr_sin_lst = [], []
error_pyr_TI_lst, error_pyr_sin_lst = [], []
amp_lst = []
for l in range(len(amp_level)):
    
    round_start_time = time.time() 

    ## Defining Saving Directories
    #########################################################################################

    SAVE_PATH_rawdata = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata):
        os.makedirs(SAVE_PATH_rawdata)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)
    
    ## Loading Data
    #######################################################################################
    start_time = time.time()
    print("Loading Raw Data for Amplitude Level %d..."%(l))
    fr_rate_sin_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_sin_fr.npy'))
    fr_rate_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'))
    fr_rate_sin_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_sin_fr.npy'))
    fr_rate_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'))
    print("Raw Data Loaded for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    
    idx_activ_pyr = fr_rate_ti_pyr>0
    if np.sum(idx_activ_pyr) > 0:
        diff_TI = fr_rate_ti_pv[idx_activ_pyr]-fr_rate_ti_pyr[idx_activ_pyr]
        diff_TI_lst.append(np.median(diff_TI))
        pyr_TI_lst.append(np.median(fr_rate_ti_pyr[idx_activ_pyr]))
        error_TI_lst.append(np.array([np.percentile(diff_TI,q=25), np.percentile(diff_TI,q=75)]))
        error_pyr_TI_lst.append(np.array([np.percentile(fr_rate_ti_pyr[idx_activ_pyr],q=25), np.percentile(fr_rate_ti_pyr[idx_activ_pyr],q=75)]))
 
        diff_sin = fr_rate_sin_pv[idx_activ_pyr]-fr_rate_sin_pyr[idx_activ_pyr]
        diff_sin_lst.append(np.median(diff_sin))
        pyr_sin_lst.append(np.median(fr_rate_sin_pyr[idx_activ_pyr]))
        error_sin_lst.append(np.array([np.percentile(diff_sin,q=25), np.percentile(diff_sin,q=75)]))   
        error_pyr_sin_lst.append(np.array([np.percentile(fr_rate_sin_pyr[idx_activ_pyr],q=25), np.percentile(fr_rate_sin_pyr[idx_activ_pyr],q=75)]))
        amp_lst.append(amp_level[l])

amp_lst = np.array(amp_lst).flatten()
diff_TI_lst, diff_sin_lst = np.array(diff_TI_lst), np.array(diff_sin_lst)
error_TI_lst, error_sin_lst = np.array(error_TI_lst).T, np.array(error_sin_lst).T

pyr_TI_lst, pyr_sin_lst = np.array(pyr_TI_lst), np.array(pyr_sin_lst)
error_pyr_TI_lst, error_pyr_sin_lst = np.array(error_pyr_TI_lst).T, np.array(error_pyr_sin_lst).T


labels = [str(int(amp_lst[i])) for i in range(len(amp_lst))]
fr_diff_medians = {'Sin':diff_sin_lst,'TI': diff_TI_lst}

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
c = 0
for attribute, measurement in fr_diff_medians.items():
    offset = width * multiplier
    if c==0:
        error = error_sin_lst
    else:
        error = error_TI_lst
    error_kw = {'elinewidth':1, 'alpha':0.5}
    rects = ax.bar(x=x+offset, yerr=error, error_kw=error_kw,capsize=3.0, height=measurement,width=width, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    c = c+1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('PV-Pyr Firing Rate (Hz)', fontsize=19)
ax.set_xlabel('Injected Current (mA)', fontsize=19)
ax.legend(loc='upper left', ncols=3, fontsize=19)
ax.set_xticks(x + width, labels)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(os.path.join(SAVE_PATH,"Fr_Diff_Bar.png"))
plt.show()

labels = [str(int(amp_lst[i])) for i in range(len(amp_lst))]
fr_diff_medians = {'Sin':pyr_sin_lst,'TI': pyr_TI_lst}

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
c = 0
for attribute, measurement in fr_diff_medians.items():
    offset = width * multiplier
    if c==0:
        error = error_pyr_sin_lst
    else:
        error = error_pyr_TI_lst
    error_kw = {'elinewidth':1, 'alpha':0.5}
    rects = ax.bar(x=x+offset, yerr=error, error_kw=error_kw,capsize=3.0, height=measurement,width=width, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1
    c = c+1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pyr Firing Rate (Hz)', fontsize=19)
ax.set_xlabel('Injected Current (mA)', fontsize=19)
ax.legend(loc='upper left', ncols=3, fontsize=19)
ax.set_xticks(x + width, labels)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig(os.path.join(SAVE_PATH,"Fr_Pyr_Bar.png"))
plt.show()

