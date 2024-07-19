import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
from elec_field import UniformField, sparse_place_rodent, ICMS, sparse_place_human
from pulse_train import PulseTrain_Sinusoid, PulseTrain_TI
import sys
import math
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
TARGET_REGION = True

### Helper Functions
##################################################################################
##################################################################################
##################################################################################
def cart_to_sph(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    r = np.sqrt(np.sum(pos**2, axis=1)).reshape(-1,1)
    theta = np.arcsin(pos[:,2].reshape(-1,1)/r).reshape(-1,1)
    phi = np.arctan2(pos[:,1],pos[:,0]).reshape(-1,1)
    sph_pos = np.hstack([r,theta,phi])
    return sph_pos
    
def sph_to_cart(pos):
    if len(pos.shape) == 1:
        pos = pos.reshape(1,-1)
    x = pos[:,0]*np.cos(pos[:,1])*np.cos(pos[:,2])
    y = pos[:,0]*np.cos(pos[:,1])*np.sin(pos[:,2])
    z = pos[:,0]*np.sin(pos[:,1])
    cart_pos = np.hstack([x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)])
    return cart_pos

def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi*(math.sqrt(5.)-1.)  # golden angle in radians
    for i in range(samples):
        y = 1-(i/float(samples-1))*2  # y goes from 1 to -1
        radius = math.sqrt(1-y*y)  # radius at y
        theta = phi*i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.array(points)

def sample_spherical(num_samples, theta_max, y_max, r=1):
    tot_samples = 0
    samples = []
    while tot_samples<num_samples:
        samples_cand = np.random.normal(loc=0, scale=1, size=(10000,3))
        samples_cand = samples_cand/np.sqrt(np.sum(samples_cand**2, axis=1)).reshape(-1,1)*r
        samples_cand = cart_to_sph(samples_cand)
        samples_cand = samples_cand[samples_cand[:,1]>theta_max]
        samples_cand = sph_to_cart(samples_cand)
        samples_cand = samples_cand[np.abs(samples_cand[:,1])<y_max]
        samples.append(samples_cand)
        tot_samples = tot_samples+samples_cand.shape[0]
    samples = np.vstack(samples)
    samples = samples[:num_samples]
    return samples


def plot_points_to_sample(coord_elec1, coord_elec2, J1, J2, freq1, freq2, points, savepath):
    
    skull_samples = np.random.normal(loc=0, scale=1, size=(10**4,3))
    skull_samples = skull_samples/np.sqrt(np.sum(skull_samples**2, axis=1)).reshape(-1,1)*(9.2-0.6)
    skull_samples = cart_to_sph(skull_samples)
    skull_samples = skull_samples[skull_samples[:,1]>(np.pi/2-7/9.2)]
    skull_samples = sph_to_cart(skull_samples)
    scalp_samples = skull_samples.copy()/(9.2-0.6)*(9.2)
    csf_samples = skull_samples.copy()/(9.2-0.6)*(9.2-1.1)
    brain_samples = skull_samples.copy()/(9.2-0.6)*(9.2-1.2)
    
    skull_samples =  skull_samples[skull_samples[:,1]<=2]
    skull_samples =  skull_samples[skull_samples[:,1]>=-2]
    
    scalp_samples =  scalp_samples[scalp_samples[:,1]<=2]
    scalp_samples =  scalp_samples[scalp_samples[:,1]>=-2]
    
    csf_samples =  csf_samples[csf_samples[:,1]<=2]
    csf_samples =  csf_samples[csf_samples[:,1]>=-2]


    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    img = ax.scatter(coord_elec1[:,0], coord_elec1[:,1], coord_elec1[:,2], linewidth=0.3, s=100, color='darkred', label=str(round(freq1))+" Hz")
    img = ax.scatter(coord_elec2[:,0], coord_elec2[:,1], coord_elec2[:,2], linewidth=0.3, s=100, color='darkgreen', label=str(round(freq2))+" Hz")
    img = ax.scatter(skull_samples[:,0], skull_samples[:,1], skull_samples[:,2], linewidth=0.3, s=10, color='grey', alpha=0.1)
    img = ax.scatter(scalp_samples[:,0], scalp_samples[:,1], scalp_samples[:,2], linewidth=0.3, s=10, color='salmon', alpha=0.1)
    img = ax.scatter(csf_samples[:,0], csf_samples[:,1], csf_samples[:,2], linewidth=0.3, s=10, color='deepskyblue', alpha=0.1)
    img = ax.scatter(brain_samples[:,0], brain_samples[:,1], brain_samples[:,2], linewidth=0.3, s=10, color='crimson',alpha=0.1)
    target_points = points[np.sum(points[:,:2]**2, axis=1)<=1]
    off_target_points = points[np.sum(points[:,:2]**2, axis=1)>=1]
    img = ax.scatter(target_points[:,0], target_points[:,1], target_points[:,2], linewidth=0.3, s=30, color='orange')
    img = ax.scatter(off_target_points[:,0], off_target_points[:,1], off_target_points[:,2], linewidth=0.3, s=30, color='blue')


    ax.set_xlabel('X-axis (cm)', fontsize=14)
    ax.set_ylabel('Y-axis (cm)', fontsize=14)
    ax.set_zlabel('Z-axis (cm)', fontsize=14)
    ax.set_title('Locations of Points Evaluated', fontsize=21)
    for i in range(coord_elec1.shape[0]):
        if J1[i]>0:
            ax.text(coord_elec1[i,0]+1.2,coord_elec1[i,1],coord_elec1[i,2]+0.2, str(round(J1[i],3)), fontsize=11)
            ax.text(coord_elec1[i,0]+2.7,coord_elec1[i,1],coord_elec1[i,2]+0.7, '2000 Hz', fontsize=11)
        else:
            ax.text(coord_elec1[i,0]+1.2,coord_elec1[i,1],coord_elec1[i,2]+0.2,str(round(J1[i],3)), fontsize=11)
    for i in range(coord_elec1.shape[0]):
        if J2[i]>0:
            ax.text(coord_elec2[i,0],coord_elec2[i,1],coord_elec2[i,2]+0.2, str(round(J2[i],3)), fontsize=11)
            ax.text(coord_elec2[i,0]-0.3,coord_elec2[i,1],coord_elec2[i,2]+0.7, '2020 Hz', fontsize=11)
        else:
            ax.text(coord_elec2[i,0],coord_elec2[i,1],coord_elec2[i,2]+0.2,str(round(J2[i],3)), fontsize=11)

    ax.text(1.2,0,9.2-0.3, 'Scalp', fontsize=11)
    ax.text(1.2,0,9.2-0.6-0.3, 'Skull', fontsize=11)
    ax.text(-2.8,-4,7.1, 'CSF', fontsize=11)
    ax.text(1.2,0,9.2-2.8, 'Brain', fontsize=11)

    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z',labelsize=12)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=np.min([xmin,ymin]), ymax=np.max([xmax,ymax]))
    ax.set_xlim(xmin=np.min([xmin,ymin]), xmax=np.max([xmax,ymax]))


    ax.view_init(25,120)
    plt.savefig(savepath+'_orientation1.png')
    ax.view_init(25,240)
    plt.savefig(savepath+'_orientation2.png')
    ax.view_init(25,90)
    plt.savefig(savepath+'_orientation3.png')
    ax.view_init(25,0)
    plt.savefig(savepath+'_orientation4.png')
    view_angle = np.linspace(0,360,361)
    def update(frame):
        ax.view_init(25,view_angle[frame])
    #ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    #ani.save(os.path.join(savepath+'.gif'), writer='pillow')
    #ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
    #plt.show()
    plt.close()

def plot_response(coord, values1, values2, y_displace, title=None, savepath=None, show=False):
    
    coord1, coord2 = coord.copy(), coord.copy()
    coord1[:,1] = coord1[:,1]-y_displace/2
    coord2[:,1] = coord2[:,1]+y_displace/2
    values = np.concatenate([values1,values2], axis=0)
    coord = np.concatenate([coord1.copy(),coord2.copy()], axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    color_map = cm.ScalarMappable(cmap='viridis_r')
    alpha_scale = values.copy()
    start_transparency = 0.5
    alpha_scale = (alpha_scale-np.min(alpha_scale))/(np.max(alpha_scale)-np.min(alpha_scale))*(1-start_transparency)+start_transparency
    rgba = color_map.to_rgba(x=values, alpha=alpha_scale, norm=True)
     
    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2],c=rgba, linewidth=2.0)
    cbar=plt.colorbar(mappable=color_map, ax=ax)
    cbar.set_label('(Hz)', fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    if title is not None:
        ax.set_title(title, fontsize=21)
    
    ax.tick_params(axis='x',which='both', left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False,labelsize=12)
    ax.tick_params(axis='y',which='both', left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False, labelsize=12)
    ax.tick_params(axis='z',which='both', left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False,labelsize=12)

    id1 = np.argmax(coord1[:,0])
    ax.text(coord1[id1,0]+3, coord1[id1,1]-2.3, coord1[id1,2], "PV\nNeuron", fontsize=13)
    id2 = np.argmax(coord2[:,0])
    ax.text(coord2[id2,0]+3, coord2[id2,1]-2.3, coord2[id2,2], "Pyr\nNeuron", fontsize=13)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=1.5*np.min([xmin,ymin]), ymax=1.5*np.max([xmax,ymax]))
    ax.set_xlim(xmin=1.5*np.min([xmin,ymin]), xmax=1.5*np.max([xmax,ymax]))

    plt.tight_layout()
    if savepath is not None:
        ax.view_init(45,120)
        plt.savefig(savepath+'_orientation1.png')
        ax.view_init(45,240)
        plt.savefig(savepath+'_orientation2.png')
        ax.view_init(45,90)
        plt.savefig(savepath+'_orientation3.png')
        ax.view_init(45,0)
        plt.savefig(savepath+'_orientation4.png')
        ax.view_init(90,0)
        plt.savefig(savepath+'_orientation5.png')

    view_angle = np.linspace(0,360,361)
    def update(frame):
        ax.view_init(45,view_angle[frame])
    #ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    #ani.save(os.path.join(savepath+'.gif'), writer='pillow')
    
    if show:
        plt.show()
    else:
        plt.close()



##################################################################################
##################################################################################
##################################################################################   

##################################################################################
######################## TI Experimental Setup ###################################
##################################################################################
SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))
if TARGET_REGION:
    SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/TI/TI_target_region')
else:
    SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/TI/TI_non-target_region')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#### Defining Electric Field Simulator
##################################################################################
##################################################################################

start_time = time.time()
print("Loading Electric Field Simulator...")
overall_radius = 9.2 ## Radius of the sphere representing the whole head
freq1, freq2 = int(sys.argv[2]), int(sys.argv[3]) ## Deciding the two frequencies for TI

#### Defing the SAVING Directory for 2elec4cm-4cm Location
SAVE_PATH = os.path.join(SAVE_PATH, '2elec4cm-4cm')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#### Defining 2000Hz Electrode Configuration
#################################################################################
theta_patch_2000 = np.pi/2-np.array([6/overall_radius,2/overall_radius]).reshape(-1,1)
phi_patch_2000 =  np.array([0,0]).reshape(-1,1)
cart_patch_2000 = sph_to_cart(np.hstack([overall_radius*np.ones([len(theta_patch_2000),1]), theta_patch_2000, phi_patch_2000]))
print("Cartesian Coordinates of Anode ([x,y,z]) for the 2000 Hz electrodes: %s cm"%(str(np.round(cart_patch_2000[1],2))))
print("Cartesian Coordinates of Cathode ([x,y,z]) for the 2000 Hz electrodes: %s cm"%(str(np.round(cart_patch_2000[0],2))))
## Describing the relative portions of current going through each electrode, the postive and negative components should sum upto 1 and -1 respectively
J_2000 = np.array([-1,1])*0.5

## Saving Electric Field Values, so we can reload them
elec_dir = os.path.join(os.getcwd(),'extracellular_voltage')
if not os.path.exists(elec_dir):
    os.makedirs(elec_dir)

file_dir_2000 = os.path.join(elec_dir,'human_elec_2elec4cm-4cm_2000kHz')
if os.path.exists(file_dir_2000+"_r.npy"):
    fname_load_2000 = file_dir_2000
    fname_save_2000 = None
else:
    fname_save_2000 = file_dir_2000
    fname_load_2000 = None
depth = 15 ## mm
elec_field_2000 = ray.put(sparse_place_human(r=depth-1.5, J=J_2000, fname_save=fname_save_2000, fname_load=fname_load_2000, theta_patch=theta_patch_2000, phi_patch=phi_patch_2000))

#### Defining 2020Hz Electrode Configuration
#################################################################################
theta_patch_2020 = np.pi/2-np.array([6/overall_radius,2/overall_radius]).reshape(-1,1)
phi_patch_2020 =  np.array([np.pi,np.pi]).reshape(-1,1)
cart_patch_2020 = sph_to_cart(np.hstack([overall_radius*np.ones([len(theta_patch_2020),1]), theta_patch_2020, phi_patch_2020]))
print("Cartesian Coordinates of Anode ([x,y,z]) for the 2020 Hz electrodes: %s cm"%(str(np.round(cart_patch_2020[1],2))))
print("Cartesian Coordinates of Cathode ([x,y,z]) for the 2020 Hz electrodes: %s cm"%(str(np.round(cart_patch_2020[0],2))))
## Describing the relative portions of current going through each electrode, the postive and negative components should sum upto 1 and -1 respectively
J_2020 = np.array([-1,1])*0.5

## Saving Electric Field Values, so we can reload them 
file_dir_2020 = os.path.join(elec_dir,'human_elec_2elec4cm-4cm_2020kHz')
if os.path.exists(file_dir_2020+"_r.npy"):
    fname_load_2020 = file_dir_2020
    fname_save_2020 = None
else:
    fname_save_2020 = file_dir_2020
    fname_load_2020 = None

elec_field_2020 = ray.put(sparse_place_human(r=depth-1.5, J=J_2020, fname_save=fname_save_2020, fname_load=fname_load_2020, theta_patch=theta_patch_2020, phi_patch=phi_patch_2020))

## Locations to evaluate neurons
if TARGET_REGION:
    points_samples = sample_spherical(num_samples=100, theta_max=np.pi/2-1/overall_radius, y_max=2, r=overall_radius-depth*10**(-1))
else:
    points_samples = sample_spherical(num_samples=250, theta_max=np.pi/2-7/overall_radius, y_max=2, r=overall_radius-depth*10**(-1))

savepath = os.path.join(SAVE_PATH,"ExpSetup")
if not os.path.exists(savepath):
    os.makedirs(savepath)
plot_points_to_sample(coord_elec1=cart_patch_2000, coord_elec2=cart_patch_2020, J1=J_2000, J2=J_2020, freq1=freq1, freq2=freq2, points=points_samples, savepath=os.path.join(savepath, "SampledPoints"))
print("Electric Field Simulator Loaded! Time Taken %s s"%(str(round(time.time()-start_time,3))))

### Defining Variables for Setting up Simulation
################################################################################

cell_id_pyr_lst =np.array([6]) #np.array([6,7,8,9,10]) ## Different Morphology for L23 Pyr Cells
cell_id_pv_lst =np.array([33]) #np.array([32,33,34,35,36]) ## Different Morphology for L23 LBC Cells
human_or_mice = ray.put(1) ## 1->mice, 0-> human
temp = ray.put(34.0) ## Celsius, temparature at which neurons are simulated
dt = ray.put(0.025) ## ms, discretization time step
num_cores = 50 ## Number of Cores used for Parallelization
SHOW_PLOTS = False ## Flag used for showing or not showing plots

#### Non-Invasive Stimulation
###################################################################################
###################################################################################

## Generating Waveforms
start_time, time_taken_round = time.time(), 0
print("Generating Waveform...")
pulse_train_sin = PulseTrain_Sinusoid()
total_time, sampling_rate = 2000, 1e6 ## ms, Hz
amp_array_2000, time_array = pulse_train_sin.amp_train(amp=1, freq=freq1, total_time=total_time, sampling_rate=sampling_rate)
amp_array_2020, _ = pulse_train_sin.amp_train(amp=1, freq=freq2, total_time=total_time, sampling_rate=sampling_rate)
amp_arra_2000, amp_array_2020 =ray.put(amp_array_2020), ray.put(amp_array_2020)
time_array = ray.put(time_array)
sampling_rate = ray.put(sampling_rate)
save_state_show = ray.put(False)
print("Waveform Generated! Time Taken %s s"%(str(round(time.time()-start_time,3))))
LOAD_DATA_FLAG = False
if not LOAD_DATA_FLAG:
    min_level, max_level = float(sys.argv[4]), float(sys.argv[5])
    amp_level = np.linspace(min_level, max_level, 8)
    np.save(os.path.join(SAVE_PATH,"Amplitude.npy"), amp_level)
    sim_already_performed = len(os.listdir(SAVE_PATH))-2
    if sim_already_performed<-1:
        sim_already_performed = input('Error Encountered while automatically detecting the number of simulations already run. Enter manually:')
    elif sim_already_performed == -1:
        sim_already_performed = 0
    if sim_already_performed != 0:
        print("Already ran %d Amplitude Levels. Starting from Amplitude Level %d"%(sim_already_performed-1, sim_already_performed))
    sim_already_performed=0
else:
    amp_level = np.load(os.path.join(SAVE_PATH,'Amplitude.npy'))
    sim_already_performed = 0

sim_already_performed = 0
for l in range(sim_already_performed,len(amp_level)):
    
    round_start_time = time.time()    
    
    #### Defining Saving Directories
    #########################################################################################
    SAVE_PATH_rawdata = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH, 'AmpLevel'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata):
        os.makedirs(SAVE_PATH_rawdata)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)


    if not LOAD_DATA_FLAG:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
        print("Starting Simulation for Amplitude Level %d"%(l))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<")
     
        split_num = int(np.floor(points_samples.shape[0]/num_cores)) ## Defines How Many Simulations Can be run in parallel

        #### Defining Locations and Orientation of Pyr and PV neurons to be evaluated
        angle = cart_to_sph(points_samples)
        angle[:,1] = np.pi/2-angle[:,1]
        angle_pyr = np.hstack([angle[:,1].copy().reshape(-1,1),angle[:,2].copy().reshape(-1,1)]) ## parameter used for specifying rotation of Pyr morphology
        angle_pv = np.hstack([angle[:,1].copy().reshape(-1,1),angle[:,2].copy().reshape(-1,1)]) ## parameter used for specifying rotation of PV morphology
        angle_pyr = np.array_split(angle_pyr, split_num, axis=0)
        angle_pv = np.array_split(angle_pv, split_num, axis=0)
        
        loc_pyr = np.array_split(points_samples*10**4, split_num, axis=0) ## cm->um, parameter used for specifying location of Pyr morphology
        loc_pv = np.array_split(points_samples*10**4, split_num, axis=0) ## cm->um, parameter used for specifying location of PV morphology
        
        ### Run Pyr Stimulation
        ######################################################################################
        cell_id_pyr = cell_id_pyr_lst[np.random.randint(len(cell_id_pyr_lst), size=points_samples.shape[0])] ## Randomly choosing a Pyr Morphology out of the 5 available
        cell_id_pyr = np.array_split(cell_id_pyr, split_num, axis=0)
        
        start_time = time.time()
        print("Simulation for Pyr Neuron Started...")
        fr_rate_ti_pyr = [] 
        for num in range(split_num):
            neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pyr[num][i], temp=temp, dt=dt, elec_field=elec_field_2000, elec_field2=elec_field_2020) for i in range(loc_pyr[num].shape[0])] ## Initializing neuron model
            ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pyr[num][i], pos_neuron=loc_pyr[num][i]) for i in range(len(neuron))]) ## Setting Extracellular Stim Paramaters
            delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation

            ## TI Stimulation
            results = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_2000, amp_array2=amp_array_2020, scale1=amp_level[l], scale2=amp_level[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores)]
            results = ray.get(results)
            fr_rate_ti_pyr.append(np.array([results[i][1] for i in range(num_cores)]).flatten())
            ## Uncomment to see plots of membrane potential
            #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
            del neuron

        fr_rate_ti_pyr = np.hstack(fr_rate_ti_pyr)
        fr_rate_ti_pyr = fr_rate_ti_pyr/(total_time*1e-03)

        print("Pyr Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))

 
        #### Run PV Stimulation
        ######################################################################################
        cell_id_pv = cell_id_pv_lst[np.random.randint(len(cell_id_pv_lst), size=points_samples.shape[0])] ## Randomly choosing a Pyr Morphology out of the 5 available
        cell_id_pv = np.array_split(cell_id_pv, split_num, axis=0)
        
        start_time = time.time()
        print("Simulation for PV Neuron Started...")
        fr_rate_ti_pv = [] 
        for num in range(split_num):
            neuron = [NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pv[num][i], temp=temp, dt=dt, elec_field=elec_field_2000, elec_field2=elec_field_2020) for i in range(loc_pyr[num].shape[0])] ## Initializing neuron model
            ray.get([neuron[i]._set_xtra_param.remote(angle=angle_pyr[num][i], pos_neuron=loc_pyr[num][i]) for i in range(len(neuron))]) ## Setting Extracellular Stim Paramaters
            delay_init, delay_final = ray.put(2000),ray.put(5) ## ms, delay added to the stimulation before and after applying stimulation
            
            ## TI Stimulation
            results = [neuron[i].stimulate.remote(time_array=time_array, amp_array=amp_array_2000, amp_array2=amp_array_2020, scale1=amp_level[l], scale2=amp_level[l], sampling_rate=sampling_rate, delay_init=delay_init, delay_final=delay_final) for i in range(num_cores)]
            results = ray.get(results)
            fr_rate_ti_pv.append(np.array([results[i][1] for i in range(num_cores)]).flatten())
            ## Uncomment to see plots of membrane potential
            #ray.get([neuron[i].plot_sim_result.remote(delay_init=delay_init) for i in range(num_cores)])
            del neuron

        fr_rate_ti_pv = np.hstack(fr_rate_ti_pv)
        fr_rate_ti_pv = fr_rate_ti_pv/(total_time*1e-03)

        print("PV Simulation Finished! Time Taken %s s"%(str(round(time.time()-start_time,3))))

        ## Saving Data
        #######################################################################################
        start_time = time.time()
        print("Saving Raw Data for Amplitude Level %d..."%(l))
        np.save(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'), fr_rate_ti_pyr)
        np.save(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'), fr_rate_ti_pv)
        print("Raw Data Saved for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    
    else:
        
        ## Loading Data
        #######################################################################################
        start_time = time.time()
        print("Loading Raw Data for Amplitude Level %d..."%(l))
        fr_rate_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata,'Pyr_ti_fr.npy'))
        fr_rate_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata,'PV_ti_fr.npy'))
        print("Raw Data Loaded for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))

    ## Plotting Results    
    #######################################################################################
    if TARGET_REGION:
        id_region = np.sqrt(np.sum(points_samples[:,:2]**2, axis=1))<=1
    else:
        id_region = np.sqrt(np.sum(points_samples[:,:2]**2, axis=1))>=1

    fr_rate_ti_pyr = fr_rate_ti_pyr[id_region] 
    fr_rate_ti_pv = fr_rate_ti_pv[id_region]
    points_samples_non_TI = points_samples[id_region]
    
    id_fr_pyr = fr_rate_ti_pyr>5

    time_taken_round = time_taken_round*(l)/(l+1)+(time.time()-round_start_time)/(l+1)
    ETA = ((len(amp_level)-l-1)*time_taken_round)/3600
    print("Simulation Finished for Amplitude Level %d! Time Taken %s hr. ETA for script to finish: %s hr"%(l, str(round((time.time()-round_start_time)/3600,3)),str(round(ETA,3))))
    
    labels = ['PV-TI', 'Pyr-TI']
    data = np.hstack([fr_rate_ti_pv[id_fr_pyr].reshape(-1,1), fr_rate_ti_pyr[id_fr_pyr].reshape(-1,1)])
    x = []
    for i in range(len(labels)):
        x.append(np.random.normal(i+1, 0.04, data.shape[0]))
    clevel = np.linspace(0,1,data.shape[1])
    plt.boxplot(data, labels=labels)
    for i in range(data.shape[1]):
        plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
    plt.title("TI Firing Rate of PV and Pyr \n at injected current %s mA"%(str(round(amp_level[l],3))), fontsize=22)
    plt.ylabel("Firing Rate (Hz)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH_plots,"Pyr_PV_ti.png"))
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    
    if TARGET_REGION:
        plot_response(coord=points_samples_non_TI, values1=fr_rate_ti_pv, values2=fr_rate_ti_pyr, y_displace=2, title=None, savepath=os.path.join(SAVE_PATH_plots,'Pyr_PV_TI_Response'), show=SHOW_PLOTS)
    else:
        plot_response(coord=points_samples_non_TI, values1=fr_rate_ti_pv, values2=fr_rate_ti_pyr, y_displace=6, title=None, savepath=os.path.join(SAVE_PATH_plots,'Pyr_PV_TI_Response'), show=SHOW_PLOTS)
