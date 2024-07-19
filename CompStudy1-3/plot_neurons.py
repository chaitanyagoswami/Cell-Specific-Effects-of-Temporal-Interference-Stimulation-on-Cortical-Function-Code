import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import time
import ray
import os
from neuron_model_parallel import NeuronSim
import math
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

### Plot Pyr Neurons
cell_id_pyr_lst = [6,7,8,9,10]
temp=34
dt=0.025
human_or_mice = 1
loc_pyr = np.array([0,0,0])
angle_pyr= np.array([0,0])
coord_lst = []

SAVE_PATH = os.path.join(os.getcwd(),'TISimResults/SimValidation/Neuron_Plots')
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

savepath = os.path.join(SAVE_PATH,'Pyr_Morphology')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for coord in coord_lst:
    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0)
ax.set_xlabel('X-axis (um)', fontsize=14)
ax.set_ylabel('Y-axis (um)', fontsize=14)
ax.set_zlabel('Z-axis (um)', fontsize=14)
ax.set_title('Morphology of Pyr Neuron Models ', fontsize=21)
#for i in range(coord_elec.shape[0]):
#    ax.text(coord_elec[i,0],coord_elec[i,1],coord_elec[i,2], 'MonoPolar Electrode')
ax.tick_params(axis='x',labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(10,120)
plt.savefig(savepath+'_orientation1.png')
ax.view_init(10,240)
plt.savefig(savepath+'_orientation2.png')
ax.view_init(10,90)
plt.savefig(savepath+'_orientation3.png')
ax.view_init(10,0)
plt.savefig(savepath+'_orientation4.png')
view_angle = np.linspace(0,360,361)
def update(frame):
    ax.view_init(10,view_angle[frame])
ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
ani.save(os.path.join(savepath+'.gif'), writer='pillow')
ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
plt.show()

### Plot PV Neurons
cell_id_pv_lst = [32,33,34,35,26]
temp=34
dt=0.025
human_or_mice = 1
loc_pv = np.array([0,0,0])
angle_pv= np.array([0,0])
coord_lst = []
for cell_id_pv in cell_id_pv_lst:
    ## Get Neuron Coordinates
    neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=cell_id_pv, temp=temp, dt=dt)
    coord = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pv, angle=angle_pv))
    coord_lst.append(coord.copy())

#displace_y_lst = list(np.linspace(-(len(cell_id_pyr_lst)-1)/2*2, (len(cell_id_pyr_lst)-1)/2*2,len(cell_id_pyr_lst))*1000)
displace_lst = [np.array([0,-1500,2000]), np.array([0,1500,2000]),np.array([0,0,0]),np.array([0,2000,0]),np.array([0,-2000,0])]
for coord, displace in zip(coord_lst, displace_lst):
    coord[:,:] = coord[:,:]+displace/2
#coord = np.vstack(coord_lst)
savepath = os.path.join(SAVE_PATH,'PV_Morphology')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for coord in coord_lst:
    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0)
ax.set_xlabel('X-axis (um)', fontsize=14)
ax.set_ylabel('Y-axis (um)', fontsize=14)
ax.set_zlabel('Z-axis (um)', fontsize=14)
ax.set_title('Morphology of PV Neuron Models', fontsize=21)
#for i in range(coord_elec.shape[0]):
#    ax.text(coord_elec[i,0],coord_elec[i,1],coord_elec[i,2], 'MonoPolar Electrode')
ax.tick_params(axis='x',labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z',labelsize=12)
ax.view_init(10,120)
plt.savefig(savepath+'_orientation1.png')
ax.view_init(10,240)
plt.savefig(savepath+'_orientation2.png')
ax.view_init(10,90)
plt.savefig(savepath+'_orientation3.png')
ax.view_init(10,0)
plt.savefig(savepath+'_orientation4.png')
view_angle = np.linspace(0,360,361)
def update(frame):
    ax.view_init(10,view_angle[frame])
ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
ani.save(os.path.join(savepath+'.gif'), writer='pillow')
ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
plt.show()

#### Plotting Electrodes and Neurons
###################################################################################
###################################################################################
savepath = os.path.join(SAVE_PATH,'MonoPolarSim')

neuron = NeuronSim.remote(human_or_mice=human_or_mice, cell_id=10, temp=temp, dt=dt)
coord_org = ray.get(neuron._translate_rotate_neuron.remote(pos_neuron=loc_pyr, angle=angle_pyr))
coord_ord = coord_org/1.2
elec_location_ICMS = fibonacci_sphere(samples=30) ## Sampling 20 approximately uniformly spaced electrode locations from a unit sphere
elec_location_ICMS = elec_location_ICMS*500
angle_pyr = np.array([0,0]) ## parameter used for specifying rotation of Pyr morphology
loc_pyr = np.array([0,0,0]) ## parameter used for specifying location of Pyr morphology
dist_lst = np.array([0.5,1,2,4,8,16])
displace_lst = [np.array([-3,0,-6]), np.array([0,0,-6]), np.array([3,0,-6]), np.array([-3,0,-3]), np.array([0,0,-3]), np.array([3,0,-3])]
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
text_lst =['0.5mm','1mm','2mm','4mm','8mm','16mm' ]
for (dist, displace, text) in zip(dist_lst, displace_lst, text_lst):
    coord_elec = elec_location_ICMS.copy()+displace*1000/2
    coord = coord_org/(dist*2)+displace*1000/2
    img = ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0, color='blue')
    img = ax.scatter(coord_elec[:,0], coord_elec[:,1], coord_elec[:,2], linewidth=0.3, s=50, color='orange', alpha=0.4)
    ax.text(displace[0]*1000/2,displace[1]*1000/2,displace[2]*1000/2+600, text)

coord_elec = elec_location_ICMS.copy()+np.array([0,0,200])
coord = coord_org/3.3+np.array([0,0,200])
ax.text(0,0,800+200, 'Uniform')
ax.quiver(coord_elec[:,0]*1.2, coord_elec[:,1]*1.2, coord_elec[:,2]*1.2, coord_elec[:,0]/1000, coord_elec[:,1]/1000, coord_elec[:,2]/1000, length=400, linewidth=1.5, color='orange')
ax.scatter(coord[:,0],coord[:,1],coord[:,2], linewidth=1.0, s=2.0, color='blue')

ax.tick_params(axis='x',which='both', left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False,labelsize=12)
ax.tick_params(axis='y',which='both',left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False, labelsize=12)
ax.tick_params(axis='z',which='both',left=False, right=False, labelleft=False, labelright=False, bottom=False,top=False,labelbottom=False,labelsize=12)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
zmin, zmax = ax.get_zlim()
ax.set_ylim(ymin=-2000, ymax=2000)
ax.set_xlim(xmin=-2000, xmax=2000)
ax.set_zlim(zmin=-4500, zmax=0)

ax.view_init(10,120)
plt.savefig(savepath+'_orientation1.png')
ax.view_init(10,240)
plt.savefig(savepath+'_orientation2.png')
ax.view_init(10,90)
plt.savefig(savepath+'_orientation3.png')
ax.view_init(10,0)
plt.savefig(savepath+'_orientation4.png')
view_angle = np.linspace(0,360,361)
def update(frame):
    ax.view_init(10,view_angle[frame])
ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
ani.save(os.path.join(savepath+'.gif'), writer='pillow')
ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
plt.show()
 

