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



def plot_response(coord, values1, values2, y_displace,c1=None,c2=None, title=None, savepath=None, show=False):
    
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
    cbar.set_label('(Hz)', fontsize=16)
    ax.set_xlabel('X-axis (cm)', fontsize=16)
    ax.set_ylabel('Y-axis (cm)', fontsize=16)
    ax.set_zlabel('Z-axis (cm)', fontsize=16)
    if title is not None:
        ax.set_title(title, fontsize=21)
    ax.tick_params(axis='x',labelsize=12)
    ax.tick_params(axis='y', labelsize=12)


    id1 = np.argmax(coord1[:,0])
    ax.text(coord1[id1,0]+1, coord1[id1,1]-0.3, coord1[id1,2], "PV\nNeuron")
    id2 = np.argmax(coord2[:,0])
    ax.text(coord2[id2,0]+1, coord2[id2,1]-0.3, coord2[id2,2], "Pyr\nNeuron")
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
    ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    ani.save(os.path.join(savepath+'.gif'), writer='pillow')
    
    if show:
        plt.show()
    else:
        plt.close()

##################################################################################
##################################################################################
##################################################################################   

##################################################################################
################## Non-Invasive Experimental Setup ###############################
##################################################################################
SEED = 1234 
np.random.seed(SEED)
print("Setting Random Seed as %s"%(str(round(SEED,3))))
cwd = os.getcwd()
print("Working in the directory: %s. All data will be saved and loaded relative to this directory"%(cwd))

SAVE_PATH_ti = os.path.join(os.getcwd(),'TISimResults/TI/TI_target_region')
if not os.path.exists(SAVE_PATH_ti):
    os.makedirs(SAVE_PATH_ti)

SAVE_PATH_no_ti = os.path.join(os.getcwd(),'TISimResults/TI/TI_non-target_region')
if not os.path.exists(SAVE_PATH_no_ti):
    os.makedirs(SAVE_PATH_no_ti)


#### Defining Electric Field Simulator
##################################################################################
##################################################################################

start_time = time.time()
print("Loading Electric Field Simulator...")
overall_radius = 9.2 ## Radius of the sphere representing the whole head

#### Defing the SAVING Directory for 2elec4cm-4cm Location
SAVE_PATH_ti = os.path.join(SAVE_PATH_ti, '2elec4cm-4cm')
if not os.path.exists(SAVE_PATH_ti):
    os.makedirs(SAVE_PATH_ti)

SAVE_PATH_no_ti = os.path.join(SAVE_PATH_no_ti, '2elec4cm-4cm')
if not os.path.exists(SAVE_PATH_no_ti):
    os.makedirs(SAVE_PATH_no_ti) 

depth = 15 ## mm

## Locations to evaluate neurons
points_samples_ti = sample_spherical(num_samples=100, theta_max=np.pi/2-1/overall_radius, y_max=2, r=overall_radius-depth*10**(-1))
points_samples = sample_spherical(num_samples=250, theta_max=np.pi/2-7/overall_radius, y_max=2, r=overall_radius-depth*10**(-1))
amp_level = np.load(os.path.join(SAVE_PATH_ti,"Amplitude.npy"))
for l in range(len(amp_level)):
    
    round_start_time = time.time() 

    ## Defining Saving Directories
    #########################################################################################
    SAVE_PATH_rawdata_ti = os.path.join(SAVE_PATH_ti, 'AmpLevel'+str(l)+'/RawData')
    SAVE_PATH_plots = os.path.join(SAVE_PATH_ti, 'AmpLevel'+str(l)+'/Plots')
    if not os.path.exists(SAVE_PATH_rawdata_ti):
        os.makedirs(SAVE_PATH_rawdata_ti)
    if not os.path.exists(SAVE_PATH_plots):
        os.makedirs(SAVE_PATH_plots)

    SAVE_PATH_rawdata_no_ti = os.path.join(SAVE_PATH_no_ti, 'AmpLevel'+str(l)+'/RawData')
    if not os.path.exists(SAVE_PATH_rawdata_no_ti):
        os.makedirs(SAVE_PATH_rawdata_no_ti)
    ## Loading Data
    #######################################################################################
    start_time = time.time()
    print("Loading Raw Data for Amplitude Level %d..."%(l))
    ti_region_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata_ti,'Pyr_ti_fr.npy'))
    ti_region_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata_ti,'PV_ti_fr.npy'))
    no_ti_region_ti_pyr = np.load(os.path.join(SAVE_PATH_rawdata_no_ti,'Pyr_ti_fr.npy'))
    no_ti_region_ti_pv = np.load(os.path.join(SAVE_PATH_rawdata_no_ti,'PV_ti_fr.npy'))
    
    idx_no_ti = np.sqrt(np.sum(points_samples[:,:2]**2, axis=1))>=1
    no_ti_region_ti_pyr = no_ti_region_ti_pyr[idx_no_ti] 
    no_ti_region_ti_pv = no_ti_region_ti_pv[idx_no_ti] 
    points_samples_no_ti = points_samples[idx_no_ti]
    
    plot_response(coord=points_samples_ti, values1=ti_region_ti_pv, values2=ti_region_ti_pyr, y_displace=2, c1=0, c2=np.max(no_ti_region_ti_pv), title=None, savepath=os.path.join(SAVE_PATH_plots,'TI_region_Pyr_PV_TI_Response'), show=True)
    plot_response(coord=points_samples_no_ti, values1=no_ti_region_ti_pv, values2=no_ti_region_ti_pyr, y_displace=6, c1=0, c2=np.max(no_ti_region_ti_pv), title=None, savepath=os.path.join(SAVE_PATH_plots,'No_TI_region_Pyr_PV_TI_Response'), show=True)

    print("Raw Data Loaded for Amplitude Level %d! Time Taken %s s"%(l,str(round(time.time()-start_time,3))))
    ## Plotting Results    
    #######################################################################################
    idx_ti = ti_region_ti_pyr>5
    labels = ['PV', 'Pyr', 'PV', 'Pyr']
    idx_no_ti = no_ti_region_ti_pyr>5 
    data = [ti_region_ti_pv[idx_ti], ti_region_ti_pyr[idx_ti], no_ti_region_ti_pv[idx_no_ti], no_ti_region_ti_pyr[idx_no_ti]]
    
    if np.sum(idx_ti)>0 and np.sum(idx_no_ti)>0:
        x = []
        x_pos = np.array([0,1,3,4])
        for i in range(len(labels)):
            x.append(np.random.normal(x_pos[i], 0.04, data[i].shape[0]))
        clevel = np.linspace(0,1,len(data))
        plt.boxplot(data, labels=labels, positions=x_pos)
        for i in range(len(data)):
            plt.scatter(x[i], data[i].flatten(), c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
        plt.ylabel("Firing Rate (Hz)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        plt.ylim(ymin=0)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH_plots,"TI_comparison.png"))
        plt.show()


