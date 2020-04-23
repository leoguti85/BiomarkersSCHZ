import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from networkx.readwrite import json_graph
import json

"""
Auxilliar functions
"""

def save_json_net(net):
	data = json_graph.node_link_data(net)	
	with open('d3j/net.json', 'w') as outfile:
		json.dump(data, outfile)
		print("Saving json netork... ")

def get_node_labels(mat_metadata,idx):
	id_node_labels  = mat_metadata['llist'][0][idx]

	node_labels = []; hemisphere_list = [];
	for i in range(0,len(id_node_labels)):
		id_label        = str(id_node_labels[i][0][0])
		hemisphere      = str(id_label[0:2])
		node_labels.append(id_label[3:])
		hemisphere_list.append(hemisphere)

	df_node_labels = pd.DataFrame(node_labels, columns=['node_label'])
	df_node_labels['hemisphere'] = hemisphere_list

	return df_node_labels


def get_connected_nodes(G_rank):

	nodes_ok = []
	n_degree = dict(G_rank.degree())
	for node in n_degree.keys():
		if n_degree[node]!=0:
			nodes_ok.append(node)

	return np.array(nodes_ok)		

def tic():
	#Homemade version of matlab tic and toc functions
	import time
	global startTime_for_tictoc
	startTime_for_tictoc = time.time()

def toc():
	import time
	if 'startTime_for_tictoc' in globals():
		print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
	else:
		print "Toc: start time not set"

def index2xyz(k,N):
	uppertriangle = (N*N + N)/2
	x = int((k+1)/uppertriangle)
	rem = (k+1) - x*uppertriangle
 
	if rem == 0:
	   return (N-1,N-1)

	for i in range(1,N+1):
		if i*N - ((i-1)*i)/2 == rem:
		   return (i-1,N-1)
		if i*N - ((i-1)*i)/2 > rem:
		   z = rem - ((i-1)*N - ((i-2)*(i-1))/2)
		   return (i-1,i+z-2) 	