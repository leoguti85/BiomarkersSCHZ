import scipy.io as sco
import numpy as np
from paths import *
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def get_max_vals(measure):

	max_index   = measure.argmax()
	max_val     = measure[max_index]
	return (max_index, max_val) 
	

resolutions = DIMS
types 	 = ['Structural','Functional','Multimodal']
labels      = ['sc_83', 'sc_129', 'sc_234', 'fc_83', 'fc_129', 'fc_234', 'mm_83', 'mm_129','mm_234'] 

X_stb = []; Y_acc = []
auc_modes = dict()

for type in types:
	
	for dim in resolutions:	

		mat1 = sco.loadmat(SAVING_MAT+'GlobalResults_'+dim+'_'+type+'_accuracy.mat')	


		if type == 'Structural':
			names_acc = 'sc_'+dim+'_mean_acc'
			names_stb = 'sc_'+dim+'_stability'
		elif type == 'Functional':	
			names_acc = 'fc_'+dim+'_mean_acc'
			names_stb = 'fc_'+dim+'_stability'
		else:
			names_acc = 'mm_'+dim+'_mean_acc'
			names_stb = 'mm_'+dim+'_stability'
			

		# first acc then stb			
		res_acc    =  get_max_vals(mat1[names_acc].ravel())
		res_stb    =  mat1[names_stb].ravel()[res_acc[0]]
		
		Y_acc.append(res_acc[1])
		X_stb.append(res_stb)	
        
		auc_modes[type+'_'+dim] = res_acc[1]*res_stb
		

df_auc = pd.DataFrame(data=auc_modes.values(), index=auc_modes.keys()).sort_values(0, ascending=False)
print("AUC")
print df_auc



fig, ax = plt.subplots()

for inx, marker in enumerate(['s', '^', '.', 'x', '+', 'v', 'o', '<', '*']):
	
	ax.plot(X_stb[inx], Y_acc[inx], marker=marker, label=labels[inx],  markersize=12, linestyle = 'None')


ax.set_xticks(X_stb, minor=True)
ax.set_yticks(Y_acc, minor=True)


ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

ax.grid(which='minor', color="navy", linestyle='dotted')


plt.legend(numpoints=1, fontsize=16)

plt.xlabel('Stability', fontsize=22)
plt.ylabel('Accuracy', fontsize=22)
plt.xlim([0.3,0.55])
plt.ylim([0.0,1.0])    

plt.show()