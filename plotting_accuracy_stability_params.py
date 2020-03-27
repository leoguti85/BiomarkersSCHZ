import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from paths import *
#plt.style.use('ggplot')

"""
Plotting accuracy and stability for different parameters. Figures will be saved in the results/ folder.
Please run this after having run mainScript_abs.py

run: python plotting_accuracy_stability.py
"""

mpl.interactive(True)
np.set_printoptions(linewidth=999999)
plt.close('all')


dims        = DIMS
types 	 = ['Structural','Functional','Multimodal']
short_types = ['sc','fc','mm']

ticks_size = 19
axis_label = 23

print(SAVING_MAT)

for resolution in dims:
	for inx, tt in enumerate(types):
		data_mat = scipy.io.loadmat(SAVING_MAT+'GlobalResults_'+resolution+'_'+tt+'_accuracy.mat')
		data_rnd = pd.read_csv(SAVING_MAT+'Random_'+resolution+'_'+tt+'_accuracy.csv', index_col=0)

		prefix = short_types[inx]+'_'+resolution+'_'
		#--------------------------------------------------------------------------------

		df_mat = pd.DataFrame(columns=['nCoeffs','rfe_step','stability','mean_acc','std_acc']) 

		df_mat['nCoeffs'] 	= data_mat[prefix+'nCoeffs'].ravel()
		df_mat['rfe_step']  = data_mat[prefix+'rfe_step'].ravel()
		df_mat['stability'] = data_mat[prefix+'stability'].ravel()
		df_mat['mean_acc']  = data_mat[prefix+'mean_acc'].ravel()
		df_mat['std_acc']   = data_mat[prefix+'std_acc'].ravel()


		rfe_step_vals    = np.array([0.2,0.5,1.0,1.0])
		rfe_step_labels  = np.array([20,50,100, 'Rnd'])	

		if tt == 'Multimodal':
			perc_values      = np.array([0.5, 1, 2, 5, 10, 25, 50])/2
		else:
			perc_values      = [0.5, 1, 2, 5, 10, 25, 50]
			


		fig = plt.figure(0, figsize=(9,15))
		for index, rfe_val in enumerate(rfe_step_vals):

			#-------------------------------------
			# Stability
			#-------------------------------------

			fig.suptitle(tt+' '+resolution+'x'+resolution)

			mask = df_mat[df_mat['rfe_step']==rfe_val]
			res_stab = mask[['nCoeffs','stability']]

			plt.subplot(212)

			if rfe_step_labels[index]=='Rnd':
				plt.plot(data_rnd['stability'].values*100, marker='v', linewidth=3.5, markersize=12.0, alpha=0.4)
			else:	
				plt.plot(res_stab['stability'].values, marker='v', linewidth=3.5, markersize=12.0)

			
			plt.xticks(range(0,res_stab.count()[0]),perc_values, fontsize=ticks_size)
			plt.yticks(fontsize=ticks_size)

			plt.xlabel("Percentage of selected features", fontsize=axis_label)			
			plt.ylabel("Kuncheva index", fontsize=axis_label)
			plt.ylim([-0.2,0.9])
			plt.legend(rfe_step_labels[0:3],  fontsize=ticks_size, loc='upper left')

			#-----------------------------------
			# Accuracy
			#----------------------------------
			res_acc = mask[['nCoeffs','mean_acc','std_acc']]

			plt.subplot(211)
			
			
			if rfe_step_labels[index]=='Rnd':
				
				plt.errorbar(x=range(0,res_acc.count()[0]), y=data_rnd['mean_acc'].values, yerr=data_rnd['std_acc'].values, marker='^', markersize=12.0, linestyle='-', linewidth=3.2, alpha=0.4)
			
			else:
				plt.errorbar(x=range(0,res_acc.count()[0]), y=res_acc['mean_acc'].values, yerr=res_acc['std_acc'].values, marker='^', linewidth=3.2, markersize=12.0)	
			
			plt.xticks(range(0,res_acc.count()[0]),perc_values, fontsize=ticks_size)
			plt.yticks(fontsize=ticks_size)

			plt.xlabel("Percentage of selected features", fontsize=axis_label)
			plt.ylabel("Accuracy", fontsize=axis_label)
			plt.legend(rfe_step_labels[0:3], fontsize=ticks_size, loc='lower left')
			plt.ylim([0,0.9])

			filename = short_types[inx]+'_'+resolution+'.png'
			plt.savefig(RESULTS+filename)
			plt.show()
   
		plt.close(fig)

