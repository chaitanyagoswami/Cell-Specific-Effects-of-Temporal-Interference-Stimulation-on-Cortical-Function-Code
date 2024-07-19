import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm

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
    
def plot_electrode_and_neuron(coord_elec, coord, savepath=None):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    img = ax.scatter(coord[:,0]*10**(-3),coord[:,1]*10**(-3),coord[:,2]*10**(-3), linewidth=1.0, s=2.0)
    img = ax.scatter(coord_elec[:,0]*10**(-3), coord_elec[:,1]*10**(-3), coord_elec[:,2]*10**(-3), linewidth=0.3, s=50)
    ax.set_xlabel('X-axis (mm)', fontsize=14)
    ax.set_ylabel('Y-axis (mm)', fontsize=14, labelpad=20)
    ax.set_zlabel('Z-axis (mm)', fontsize=14, labelpad=20)
    ax.set_title('Neuron Orientation w.r.t Electrode', fontsize=21)
    ax.tick_params(axis='x',which='both',labelsize=12)
    ax.tick_params(axis='y',which='both',labelsize=12, pad=10)
    ax.tick_params(axis='z',which='both',labelsize=12, pad=10)
    
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
        ax.view_init(10,view_angle[frame])
    ani = animation.FuncAnimation(fig=fig, func=update, frames=361, interval=20)
    ani.save(os.path.join(savepath+'.gif'), writer='pillow')
    ani.save(os.path.join(savepath+'.mp4'), writer='ffmpeg')
    plt.show()
