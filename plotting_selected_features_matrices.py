import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from paths import * 

"""
Plotting matrices of selected features.
Run as
python plotting_selected_features_matrices.py

"""

mpl.interactive(True)
np.set_printoptions(linewidth=999999)
plt.close('all')
plt.style.use('ggplot')


def plot_many_matrices(y, m_code_prefix='A', tit='Selected Features'):
	'''
	Input:  rank_matrices, m_code_prefix: 'A', 'A_sc', 'A_fc'
	Output: Plot rank matrices for all combination of parameters
	'''

	perc_feats_keep = np.array([0.5,1,2,5,10, 25, 50])
	rfe_step_range = np.array([20,50,100]) 	
	tits = list(itertools.product(perc_feats_keep,rfe_step_range))


	fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(30,16))

	plt.subplots_adjust(hspace=0.05)


	i=0
	for ax in axes.flat:
		M = y[0][m_code_prefix+str(i+1)][0]
		cax = ax.imshow(M,interpolation='nearest')
		ax.set_title('selected: '+str(tits[i][0])+'% \nstep size: '+str(tits[i][1])+'%',  fontsize=20)
		ax.grid(False)
		i+=1

		if i == len(y[0][0]):
			break;

	fig.suptitle("Selected Features, "+tit, fontsize=18)
	plt.savefig(RESULTS+'m_'+tit+'.png')
	plt.close(fig)


#------------------------------------------------------------------------------------

types = ['Structural', 'Functional', 'Multimodal']

dims   = DIMS

for connectivity in types:

	for resolution in dims:
		
		data_mat =  scipy.io.loadmat(SAVING_MAT+'GlobalResults_'+resolution+'_'+connectivity+'_accuracy.mat')

		if connectivity=="Structural":
			prefix = 'sc_'+resolution

				
		elif connectivity=="Functional":
			prefix = 'fc_'+resolution

		elif connectivity=="Multimodal":
			prefix = 'mm_'+resolution
			
		rank_matrices  = data_mat[prefix+'_'+'Matrices']

		print(SAVING_MAT);print(connectivity);print(resolution);print(prefix);print('\n')

		if connectivity=="Multimodal":
			plot_many_matrices(rank_matrices, 'A_sc', 'sc_'+prefix)
			plot_many_matrices(rank_matrices, 'A_fc', 'fc_'+prefix)
		else:	
			plot_many_matrices(rank_matrices, 'A', prefix)
