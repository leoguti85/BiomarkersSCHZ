import numpy as np
import scipy.io
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, make_scorer
from feature_selection_stability import Kuncheva_index
from sklearn.pipeline import Pipeline
import argparse
import pandas as pd
from paths import * 
from functions import * 
"""
connectivity: Structural, Functional, Multimodal
resolutions:  83 , 129,  234

example,
python mainScript_abs.py -connectivity Structural -resolution 83

"""

subjects_indx    = pd.read_csv(SELECTED_SUBJECTS27, header=None)
recall_scorer    = make_scorer(recall_score,average='macro') # recall__score with average=macro is the balanced accuracy

"""
Auxiliar functions
"""
def get_attrib(attrib):
	selected_features = []
	for k in bestFeatures.keys():
		selected_features.append(bestFeatures[k][attrib])
		
	return np.array(selected_features)	

def computing_weights_mat(svm_weights, selected_var):
	svm_matrix = np.zeros((int(resolution),int(resolution)))
	for i in range(0,len(selected_var)):
		
			row, col = index2xyz(selected_var[i], int(resolution)) 	# getting  index (x,y) of the feature k
			svm_matrix[row, col]  = svm_weights[i]
			svm_matrix[col, row] = svm_weights[i]

	return svm_matrix

def computing_ranking_mat(selected_var):
	ranking_matrix = np.zeros((int(resolution),int(resolution)))
	for i in range(0,selected_var.shape[0]):
		for j in range(selected_var.shape[1]):
			
			row, col = index2xyz(selected_var[i,j], int(resolution)) 	# getting  index (x,y) of the feature k
			ranking_matrix[row,col]+=1
			ranking_matrix[col, row] = ranking_matrix[row,col]

	return ranking_matrix	

def computing_matrix_max_acc(sel_var):
	ranking_matrix = np.zeros((int(resolution),int(resolution)))
	for i in range(0,len(sel_var)):

		row, col = index2xyz(sel_var[i], int(resolution)) 	# getting  index (x,y) of the feature k
		ranking_matrix[row, col]  = 1
		ranking_matrix[col, row] = 1

	return ranking_matrix		

def computing_matrix_max_multi_acc(selected_var):
	ranking_matrix_sc = np.zeros((int(resolution),int(resolution)))
	ranking_matrix_fc = np.zeros((int(resolution),int(resolution)))
	max_num_features = int(resolution)*(int(resolution)+1)/2
	
	for i in range(0,len(selected_var)):
		
			row, col = index2xyz(selected_var[i], int(resolution)) 	# getting  index (x,y) of the feature k

			if selected_var[i] < max_num_features:
				ranking_matrix_sc[row, col] = 1
				ranking_matrix_sc[col, row] = 1
			else:
				ranking_matrix_fc[row,col]  = 1	
				ranking_matrix_fc[col, row] = 1

	return (ranking_matrix_sc,ranking_matrix_fc)

def computing_multimodal_ranking_mat(selected_var):
	ranking_matrix_sc = np.zeros((int(resolution),int(resolution)))
	ranking_matrix_fc = np.zeros((int(resolution),int(resolution)))
	max_num_features = int(resolution)*(int(resolution)+1)/2
	
	for i in range(0,selected_var.shape[0]):
		for j in range(selected_var.shape[1]):

			row, col = index2xyz(selected_var[i,j], int(resolution)) 	# getting  index (x,y) of the feature k

			if selected_var[i,j] < max_num_features:
				ranking_matrix_sc[row,col]+=1
				ranking_matrix_sc[col, row] = ranking_matrix_sc[row,col]
			else:
				ranking_matrix_fc[row,col]+=1	
				ranking_matrix_fc[col, row] = ranking_matrix_fc[row,col]

	return (ranking_matrix_sc,ranking_matrix_fc)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-connectivity', dest='connectivity', help='structural or functional')
	parser.add_argument('-resolution', dest='resolution', help='dimension')		
	return parser.parse_args()

#-------------------------------------------------------------------	

"""
Defining parameters

connectivity: Structural, Functional, Multimodal
resolutions:  83 , 129,  234

example,
python mainScript_abs.py -connectivity Structural -resolution 83
"""
args         = parse_args()

connectivity = args.connectivity   # "Structural", "Functional" or "Multimodal"; case sensitive
resolution   = args.resolution     # 83 , 129,  234
scoring      = 'accuracy'          # 'accuracy' or 'balanced_accuracy'

if scoring=='accuracy':
	scoring_fct = 'accuracy'
elif scoring=='balanced_accuracy':
	scoring_fct = recall_scorer
else:
	print("Wrong 'scoring' value; 'accuracy' is used by default")
	scoring_fct = 'accuracy'


"""
Loading data from precomputed Matlab files (vectorized version of the upper triangular connectivity matrices)
"""

if connectivity=="Structural":
	features  = scipy.io.loadmat(SC_FEATURES+'features_BoE_SC_'+resolution+'.mat') # Dictionary structure
	features  = features['features'] # Get rid of header
	features  = features.transpose()
	prefix = 'sc_'+resolution+'_'
elif connectivity=="Functional":
	#features  = scipy.io.loadmat('FC/features_BoE_FC_thres_'+resolution+'_0.mat') # Dictionary structure
	features  = scipy.io.loadmat(FC_FEATURES+'features_BoE_FC_'+resolution+'.mat') # Dictionary structure
	features  = features['features'] # Get rid of header
	features  = features.transpose()
	prefix = 'fc_'+resolution+'_'
elif connectivity=="Multimodal":
	# STRUCTURAL CONNECTIVITY
	featuresSC = scipy.io.loadmat(SC_FEATURES+'features_BoE_SC_'+resolution+'.mat') # Dictionary structure
	featuresSC = featuresSC['features'] # Get rid of header
	featuresSC = featuresSC.transpose()
	# FUNCTIONAL CONNECTIVITY
	#featuresFC = scipy.io.loadmat('FC/features_BoE_FC_thres_'+resolution+'_0.mat') # Dictionary structure
	featuresFC  = scipy.io.loadmat(FC_FEATURES+'features_BoE_FC_'+resolution+'.mat') # Dictionary structure
	featuresFC = featuresFC['features'] # Get rid of header
	featuresFC = featuresFC.transpose()
	# Concatenation of both modalities
	features   = np.concatenate([featuresSC,featuresFC],1)
	prefix = 'mm_'+resolution+'_'
else:
	print("Wrong 'connectivity' argument ! \"Functional\" is used by default")
	features = scipy.io.loadmat('features_BoE_FC_thres_'+resolution+'_0.mat') # Dictionary structure
	features = features['features'] # Get rid of header
	features = features.transpose()
"""
Define class labels
"""
print prefix
print(connectivity)
print(resolution)

features = features[subjects_indx.values.ravel()]

labels = np.ones(27+27)
for i in range(27,labels.shape[0]):
	labels[i] = 0 # -1 is for SCHZ subjects

"""
Parameters
"""

ext_folds      = 5
inner_folds    = 5
params_C 	   = [1e-02,0.1,1.,10.,100.]

#------------------------------------------------------------------------------------------------------------
seed_range     = range(1,21)  # Random seed for the StratifiedKFolding, i.e., reptitions of the outer CV				

if connectivity=="Multimodal":
   perc_feats_keep = np.array([0.005,0.01,0.02,0.05,0.1, 0.25, 0.5])/2.0
else:
   perc_feats_keep = np.array([0.005,0.01,0.02,0.05,0.1, 0.25, 0.5])  


rfe_step_range = np.array([0.2,0.5,features.shape[1]]) 	# Percentage of features eliminated at each iteration


param_grid = dict(fs__estimator__C=params_C)
nCoeffs_range = np.floor(features.shape[1]*perc_feats_keep).astype('int')
#------------------------------------------------------------------------------------------------------------
"""
MAIN LOOP
"""
tic();
results      = np.zeros((ext_folds,len(seed_range)))
bestFeatures = dict()

sample     = 0
count_iter = 0
for seed_idx, seed in enumerate(seed_range):

	
	skf = StratifiedKFold(n_splits=ext_folds, shuffle=True, random_state=seed)
	
	for nCoeffs_idx, nCoeffs in enumerate(nCoeffs_range): #features to keep

		for rfe_step_idx, rfe_step in enumerate(rfe_step_range):


			print(str(count_iter)+'/'+str(len(seed_range)*len(nCoeffs_range)*len(rfe_step_range)))

			for train_index, test_index in skf.split(features, labels): # external CV

				X_train, X_test = features[train_index], features[test_index]
				y_train, y_test = labels[train_index], labels[test_index]

				scaler = MinMaxScaler()
				sv     = LinearSVC()
				rfe    = RFE(sv, step=rfe_step, n_features_to_select=nCoeffs)

				# Defining scaler + rfe
				pipe = Pipeline([('std_scaler', scaler), ('fs', rfe)])

				clf = GridSearchCV(pipe, param_grid=param_grid, cv=inner_folds, scoring=scoring_fct, n_jobs=6)
				y_score = clf.fit(X_train, y_train)

				#print(clf.best_params_) 
							
				best_model = clf.best_estimator_
				selector   = best_model.named_steps['fs']

				y_true, y_pred = y_test, clf.predict(X_test)   
				acc = accuracy_score(y_true, y_pred)
				
				data = {'seed': seed, 'nCoeffs':     nCoeffs, 'perc_selected_feats': perc_feats_keep[nCoeffs_idx], 
									  'rfe_step':    rfe_step, 'accuracy': acc,
									  'features':    selector.support_.nonzero()[0].astype('int'),
									  'best_score':  clf.best_score_,
									  'svm_weights': selector.estimator_.coef_.ravel()}
									  
				bestFeatures[sample] = data
				sample+=1
    
			count_iter+=1 # for displaying	

toc();

df = pd.DataFrame() # each row is a sample

df['seed'] 					= get_attrib('seed') # uses bestFeatures 
df['nCoeffs'] 				= get_attrib('nCoeffs')
df['perc_selected_feats'] 	     = get_attrib('perc_selected_feats')
df['rfe_step'] 				= get_attrib('rfe_step')
df['accuracy'] 				= get_attrib('accuracy')
df['best_score'] 			= get_attrib('best_score')
features_attr 				= get_attrib('features')
#features_rankings 			= get_attrib('rankings')
svm_weights 				= get_attrib('svm_weights')

"""
Computing stability 
"""
df_results = pd.DataFrame(columns=['nCoeffs','rfe_step','stability','mean_acc','std_acc','best_score']) 

# Write to .mat matrices
data_mats     = dict()
weights_mat   = dict()
intercept_mat = dict()
indx_key      = 0 

for pfs in perc_feats_keep: 		# nCoeff
	for rfes in rfe_step_range:		# rfe_step

		res = df[(df['perc_selected_feats']==pfs) & (df['rfe_step']==rfes)] # there are seeds x cv_ext samples (real number of samples)
		selected_var = np.vstack(features_attr[res.index])
		stab = Kuncheva_index(selected_var,features.shape[1])

		if rfes == features.shape[1]:
			rfes = 1.0
		row = [pfs, rfes, stab,res['accuracy'].mean(), res['accuracy'].std(), res['best_score']]
		df_results.loc[indx_key] = row
		
		max_idx = res.idxmax(axis=0)['accuracy'] 		# select features which gave the best accuracy
		
		if connectivity=='Multimodal':

			A_sc, A_fc = computing_multimodal_ranking_mat(selected_var)
						
			data_mats['A_sc'+str(indx_key+1)] = A_sc
			data_mats['A_fc'+str(indx_key+1)] = A_fc
		else:	
			rank_mat = computing_ranking_mat(selected_var)
						
			data_mats['A'+str(indx_key+1)] 		= rank_mat

		weights_mat['svm_w'+str(indx_key+1)] = computing_weights_mat(svm_weights[max_idx], features_attr[max_idx])		
		indx_key+=1

		

"""
Preparing data structure to save Matlab format file
"""

data1 = {prefix+'nCoeffs': 		df_results['nCoeffs'].values, 
		 prefix+'rfe_step':	    df_results['rfe_step'].values, 
		 prefix+'stability':    df_results['stability'].values, 
		 prefix+'mean_acc':     df_results['mean_acc'].values, 
		 prefix+'std_acc':      df_results['std_acc'].values, 
		 prefix+'Matrices':     data_mats,
		 prefix+'svm_weights':  weights_mat}

#scipy.io.savemat(SAVING_MAT+'GlobalResults_'+resolution+'_'+connectivity+'_'+scoring+'.mat', data1); print("Saving .mat...");

