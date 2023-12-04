import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

LOAD_PATH = os.path.join(os.getcwd(),'TISimResults/MonopolarSim')
filenames = ['Results_distance500.0um', 'Results_distance1000.0um', 'Results_distance2000.0um', 'Results_distance4000.0um', 'Results_distance8000.0um', 'Results_distance16000.0um', 'Results_uniform']
files_dir = [os.path.join(LOAD_PATH,file) for file in filenames if file[:6]=='Result']
activ_thresh_pyr, activ_thresh_pv  = [], []
fr_activ_thresh_pyr, fr_activ_thresh_pv  = [], []
labels = []
for file in files_dir:
    activ_thresh_pyr.append(np.load(os.path.join(file, 'activation_Pyr.npy')))
    activ_thresh_pv.append(np.load(os.path.join(file, 'activation_PV.npy')))
    fr_activ_thresh_pyr.append(np.load(os.path.join(file, 'activation_fr_Pyr.npy')))
    fr_activ_thresh_pv.append(np.load(os.path.join(file, 'activation_fr_PV.npy')))

activ_thresh_pyr, activ_thresh_pv = np.array(activ_thresh_pyr), np.array(activ_thresh_pv)
fr_activ_thresh_pyr, fr_activ_thresh_pv = np.array(fr_activ_thresh_pyr), np.array(fr_activ_thresh_pv)

#### Calculatin average statistics
activ_thresh_pyr_median, activ_thresh_pv_median = np.median(activ_thresh_pyr, axis=1), np.median(activ_thresh_pv, axis=1)
activ_thresh_pyr_mean, activ_thresh_pv_mean = np.mean(activ_thresh_pyr, axis=1), np.mean(activ_thresh_pv, axis=1)
activ_inc = (activ_thresh_pyr-activ_thresh_pv)/np.median(activ_thresh_pv,axis=1).reshape(-1,1)*100
percentage = np.sum(activ_inc[4:]<0)/(np.sum(activ_inc[4:]>=0)+np.sum(activ_inc[4:]<0))*100
print(percentage)
print((np.sum(activ_inc>=0)+np.sum(activ_inc<0)))
exit()
activ_inc_median = np.median(activ_inc, axis=1)
activ_inc_mean = np.mean(activ_inc, axis=1)
activ_inc_std = np.sqrt(np.var(activ_inc, axis=1))
activ_inc_25 = np.percentile(activ_inc,q=25, axis=1)
activ_inc_75 = np.percentile(activ_inc,q=75, axis=1)

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = np.median(activ_inc, axis=1) 
bar_container = ax.bar(labels, data)
ax.set_ylabel('% increase threshold', fontsize=19)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,'Median_Percentage_Increase.png'))
plt.show()

fig, ax = plt.subplots()
data = np.mean(activ_inc, axis=1)
bar_container = ax.bar(labels, data)
ax.set_ylabel('% increase threshold', fontsize=19)
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,'Mean_Percentage_Increase.png'))
plt.show()

fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = activ_inc.T
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
bp = plt.boxplot(data, labels=labels)
plt.axhline(xmin=0, xmax=1,y=0, color='black', linestyle='--')

for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c='C0', alpha=0.4)
plt.ylabel("% increase threshold", fontsize=20)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"Paired_Median_Percentage_Increase.png"))
plt.show()



fig, ax = plt.subplots()
labels = ['Sin','TI']*7
data = []
for i in range(7):
    data.append(fr_activ_thresh_pyr[i,:,2]-fr_activ_thresh_pyr[i,:,0])
    data.append(fr_activ_thresh_pyr[i,:,3]-fr_activ_thresh_pyr[i,:,1])
x_pos = np.array([0,1,3,4,6,7,9,10,12,13,15,16,18,19])
x = []
for i in range(len(x_pos)):
    x.append(np.random.normal(x_pos[i], 0.04, data[i].shape[0]))
bp = plt.boxplot(data, labels=labels,positions=x_pos)
for i in range(len(x_pos)):
    if i%2==0:
        if i==0:
            plt.scatter(x[i], data[i].flatten(), c='C0', alpha=0.4, label='Pure')
        else:
            plt.scatter(x[i], data[i].flatten(), c='C0', alpha=0.4)
    else:
        if i==1:
            plt.scatter(x[i], data[i].flatten(), c='C1', alpha=0.4, label='Modulated')
        else:
            plt.scatter(x[i], data[i].flatten(), c='C1', alpha=0.4)

#plt.title('% increase between Activation \n Thresholds of Pyr and PV', fontsize=22)
plt.ylabel("% increase threshold", fontsize=20)
plt.legend(fontsize=18, ncols=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"Paired_Median_Percentage_FR_Diff.png"))
plt.show()




fig, ax = plt.subplots()
labels = ['0.5mm', '1mm', '2mm', '4mm', '8mm', '16mm', 'Unif']
data = fr_activ_thresh_pyr[:,:,0]-fr_activ_thresh_pyr[:,:,1]
data = data.T
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
bp = plt.boxplot(data, labels=labels)
xlims = ax.get_xlim()
plt.hlines(0,xmin=xlims[0], xmax=xlims[1], linestyle='--', color='black')
#
#xtickslocs = ax.get_xticks()
#ymin, _ = ax.get_ylim()
#print('xticks pixel coordinates')
#xloc_lst = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
#for (medline,xloc) in zip(bp['medians'],xloc_lst):
#    linedata = medline.get_ydata()
#    median = linedata[0]
#    ax.text(xloc[0], median, str(round(median,2)))
#
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
#plt.title('Diff. between Pyr Firing rates of TI \n and Sin at Pyr Activation Threshold', fontsize=22)
plt.ylabel("Pure - Mod. Sin Fr. Rate", fontsize=18)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"Pyr_fr_Diff.png"))
plt.show()


fig, ax = plt.subplots()
labels = ['0.5\nmm', '1\nmm', '2\nmm', '4\nmm', '8\nmm', '16\nmm', 'Unif']
data = fr_activ_thresh_pv[:,:,2]-fr_activ_thresh_pv[:,:,3]
data = data.T
x = []
for i in range(len(labels)):
    x.append(np.random.normal(i+1, 0.04, data.shape[0]))
clevel = np.linspace(0,1,data.shape[1])
bp = plt.boxplot(data, labels=labels)
xlims = ax.get_xlim()
plt.hlines(0,xmin=xlims[0], xmax=xlims[1], linestyle='--', color='black')
#xtickslocs = ax.get_xticks()
#ymin, _ = ax.get_ylim()
#print('xticks pixel coordinates')
#xloc_lst = ax.transData.transform([(xtick, ymin) for xtick in xtickslocs])
#for (medline,xloc) in zip(bp['medians'],xloc_lst):
#    linedata = medline.get_ydata()
#    median = linedata[0]
#    ax.text(xloc[0], median, str(round(median,2)))
#
for i in range(data.shape[1]):
    plt.scatter(x[i], data[:,i], c=np.array(cm.prism(clevel[i])).reshape(1,-1), alpha=0.4)
#plt.title('Diff. between PV Firing rates of TI \n and Sin at PV Activation Threshold', fontsize=22)
plt.ylabel("Pure - Mod. Sin Fr. Rate", fontsize=18)
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"PV_fr_Diff.png"))
plt.show()


labels = ("0.5\nmm", "1\nmm", "2\nmm", "4\nmm", "8\nmm", "16\nmm", "Unif")
data = np.vstack([np.median(fr_activ_thresh_pyr[:,:,2]-fr_activ_thresh_pyr[:,:,0], axis=1).reshape(1,-1),np.median(fr_activ_thresh_pyr[:,:,3]-fr_activ_thresh_pyr[:,:,1], axis=1).reshape(1,-1)])
print(data.shape)
fr_diff_medians = {'Sin':data[0],'TI': data[1]}

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in fr_diff_medians.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('PV-Pyr Firing Rate (Hz)', fontsize=19)
ax.legend(loc='upper right', ncols=3, fontsize=19)
ax.set_xticks(x + width, labels)
ax.tick_params(axis='x', labelsize=19)
ax.tick_params(axis='y', labelsize=19)


plt.tight_layout()
plt.savefig(os.path.join(LOAD_PATH,"PV_fr_Diff_Bar.png"))
plt.show()
