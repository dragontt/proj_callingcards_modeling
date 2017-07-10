#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster


def cluster_gnashyframe(gnashyframe,distance_cutoff):
	"""This function takes a gnashy dataframe as input and clusters the insertions
	by hierarchical clustering.  It uses euclidean distance as the metric and seeks to minimize
	the average pairwise distance of cluster members.  The distance_cutoff parameter is used to define
	the different clusters.  Because hierarchical clustering is computationally inefficient, it breaks
	the chromosomes up into different pieces and clusters the individual pieces.  To ensure that the breakpoints
	don't break up potentia cluster, it searches for regions of the chromosome with a large distance between
	insertions.  If it cannot find a distance that is greater than the distance cutoff, then it prints a 
	warning.  The program outputs a frame of clusters with the following columns: 
	["Chr","Start","End","Center","Experiment Hops","Fraction Experiment"]"""

	MAX_ARRAY_SIZE = 5000  #maximum number of insertions to cluster
	MIN_JUMP = 1000 #minimum number of insertions to cluster

	cluster_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","Experiment Hops","Fraction Experiment"])

	grouped = gnashyframe.groupby(['Chr']) #group insertions by chromosome
	
	for chro in grouped.groups:
		print "Analyzing chromosome "+str(chro)
		full_hop_list = list(grouped.get_group(chro)['Hop Pos'])
		#find gap to split chromosome for clustering
		next_index = 0
		while next_index < len(full_hop_list):
			if (len(full_hop_list)-next_index) < MAX_ARRAY_SIZE:
				hop_list = full_hop_list[next_index:len(full_hop_list)]
				hop_list = [[x] for x in hop_list]
				old_index = next_index
				next_index = len(full_hop_list)
			else:
				gap_list = full_hop_list[next_index+MIN_JUMP:next_index+MAX_ARRAY_SIZE]
				difference_list = [j-i for i, j in zip(gap_list[:-1], gap_list[1:])]
				max_dist = max(difference_list)
				gap_index = difference_list.index(max(difference_list))+1
				hop_list = full_hop_list[next_index:next_index+MIN_JUMP+gap_index]
				hop_list = [[x] for x in hop_list]
				old_index = next_index
				next_index = next_index+MIN_JUMP+gap_index
				if max_dist < distance_cutoff:
					print "Warning. Chromosome "+str(chro)+" "+hop_list[0]+" jump distance of "+\
					max_dist+" is less than distance_cutoff of "+distance_cutoff+"."
				print old_index, next_index-1, max_dist
			#cluster split chromosome
			Z = linkage(hop_list,'average','euclidean')
			#find clusters
			clusters = fcluster(Z,distance_cutoff,criterion='distance')
			#record clusters
			pos_clus_frame = pd.DataFrame(columns=['Hop Position','Cluster'])
			pos_clus_frame['Hop Position'] = hop_list
			pos_clus_frame['Cluster'] = clusters
			grouped_hops = pos_clus_frame.groupby(['Cluster'])
			for cluster in grouped_hops.groups:
				hops_in_cluster = list(grouped_hops.get_group(cluster)['Hop Position'])
				chromosome = chro
				start = min(hops_in_cluster)[0]
				end = max(hops_in_cluster)[0]
				center = np.median(hops_in_cluster)
				experiment_hops = len(hops_in_cluster) 
				fraction_experiment = float(experiment_hops)/len(gnashyframe)
				cluster_frame = cluster_frame.append(pd.DataFrame({"Chr":[chromosome],"Start":[start],
					"End":[end],"Center":[center],"Experiment Hops":[experiment_hops],
					"Fraction Experiment":[fraction_experiment]}))
	#sort cluster frame by chr then position
	cluster_frame = cluster_frame.sort_values(["Chr","Start"])
	cluster_frame = cluster_frame[["Chr","Start","End","Center","Experiment Hops","Fraction Experiment"]]
	return cluster_frame


def cluster_orf2hops(orf2hops_frame, bg_frame, expt_totals, bg_totals, distance_cutoff):
	"""This function takes a orf2hops dataframe as input and clusters the insertions
	by hierarchical clustering.  It uses euclidean distance as the metric and seeks to minimize
	the average pairwise distance of cluster members.  The distance_cutoff parameter is used to define
	the different clusters.  Because hierarchical clustering is computationally inefficient, it breaks
	the chromosomes up into different pieces and clusters the individual pieces.  To ensure that the breakpoints
	don't break up potentia cluster, it searches for regions of the chromosome with a large distance between
	insertions.  If it cannot find a distance that is greater than the distance cutoff, then it prints a 
	warning.  The program outputs a frame of clusters with the following columns: 
	[]"""

	MAX_ARRAY_SIZE = 5000  #maximum number of insertions to cluster
	MIN_JUMP = 1000 #minimum number of insertions to cluster

	cluster_frame = pd.DataFrame(columns = ["Orf","Start","End","Center","Exp_TPH","Exp_RPH","Bg_TPH","Bg_RPH","Exp_TPH_BS","Exp_RPH_BS","Strand","Dist_to_ATG"])

	## get totals hops and reads for experiment and background data
	total_exp_hops = expt_totals['hops']
	total_exp_reads = expt_totals['reads']
	total_bg_hops = bg_totals['hops']
	total_bg_reads = bg_totals['reads']

	grouped = orf2hops_frame.groupby(['Orf']) #group experiment insertions by orf
	orf_counter = 0

	## perform clustering on each orf promoter
	for orf in grouped.groups:
		orf_counter += 1
		if orf_counter % 1000 == 0:
			print "Analyzing "+str(orf_counter)+"th orf"
		
		full_hop_list = list(grouped.get_group(orf)['Hop_pos'])
		#find gap to split orf promoter for clustering
		next_index = 0
		while next_index < len(full_hop_list):

			if (len(full_hop_list)-next_index) < MAX_ARRAY_SIZE:
				hop_list = full_hop_list[next_index:len(full_hop_list)]
				hop_list = [[x] for x in hop_list]
				old_index = next_index
				next_index = len(full_hop_list)
			else:
				gap_list = full_hop_list[next_index+MIN_JUMP:next_index+MAX_ARRAY_SIZE]
				difference_list = [j-i for i, j in zip(gap_list[:-1], gap_list[1:])]
				max_dist = max(difference_list)
				gap_index = difference_list.index(max(difference_list))+1
				hop_list = full_hop_list[next_index:next_index+MIN_JUMP+gap_index]
				hop_list = [[x] for x in hop_list]
				old_index = next_index
				next_index = next_index+MIN_JUMP+gap_index
				if max_dist < distance_cutoff:
					print "Warning. orf "+str(orf)+" "+hop_list[0]+" jump distance of "+\
					max_dist+" is less than distance_cutoff of "+distance_cutoff+"."
				print old_index, next_index-1, max_dist

			##check for the need of clustering
			if len(hop_list) == 1:
				clusters = [1]
			else:
				#cluster split orf pormoter
				Z = linkage(hop_list,'average','euclidean')
				#find clusters
				clusters = fcluster(Z,distance_cutoff,criterion='distance')

			#record clusters
			orf_indx = np.where(orf2hops_frame['Orf'] == orf)[0]
			reads_arr = [x for x in orf2hops_frame['Reads'][orf_indx]]
			orf_atg = orf2hops_frame['Orf_pos'][orf_indx[0]]
			orf_strand = orf2hops_frame['Strand'][orf_indx[0]]

			pos_clus_frame = pd.DataFrame(columns=['Hop_ps','Cluster'])
			pos_clus_frame['Hop_pos'] = hop_list
			pos_clus_frame['Cluster'] = clusters
			grouped_hops = pos_clus_frame.groupby(['Cluster'])

			for cluster in grouped_hops.groups:
				hops_in_cluster = list(grouped_hops.get_group(cluster)['Hop_pos'])
				start = min(hops_in_cluster)[0]
				end = max(hops_in_cluster)[0]
				center = np.median(hops_in_cluster)

				## get foreground (experiment)
				exp_hops = len(hops_in_cluster) 
				exp_TPH = float(exp_hops)/total_exp_hops*100000
				exp_reads = sum(reads_arr)
				exp_RPH = float(exp_reads)/total_exp_reads*100000

				## get background 
				bg_frame_orf = bg_frame.loc[bg_frame["Orf"]==orf]
				bg_hops = []
				bg_reads = []
				for k,row in bg_frame_orf.iterrows():
					if row["Hop_pos"] >= start and row["Hop_pos"] <= end:
						bg_hops.append(row["Hop_pos"])
						bg_reads.append(row["Reads"])
				bg_TPH = float(len(bg_hops))/total_bg_hops*100000
				bg_RPH = float(sum(bg_reads))/total_bg_reads*100000

				## background subtraction
				exp_TPH_BS = exp_TPH - bg_TPH if exp_TPH > bg_TPH else 0
				exp_RPH_BS = exp_RPH - bg_RPH if exp_RPH > bg_RPH else 0

				## calcuate distance of peak center to ATG
				dist_to_atg = center - orf_atg if orf_strand=='+' else orf_atg - center

				cluster_frame = cluster_frame.append(pd.DataFrame({"Orf":[orf],
							"Start":[start],"End":[end],"Center":[center],
							"Exp_TPH":[exp_TPH],"Exp_RPH":[exp_RPH],
							"Bg_TPH":[bg_TPH],"Bg_RPH":[bg_RPH],
							"Exp_TPH_BS":[exp_TPH_BS],"Exp_RPH_BS":[exp_RPH_BS],
							"Strand":[orf_strand],"Dist_to_ATG":[dist_to_atg]}))
	
	#sort cluster frame by orf then position
	cluster_frame = cluster_frame.sort_values(["Orf","Start"])
	cluster_frame = cluster_frame[["Orf","Start","End","Center","Exp_TPH","Exp_RPH","Bg_TPH","Bg_RPH","Exp_TPH_BS","Exp_RPH_BS","Strand","Dist_to_ATG"]]
	
	return cluster_frame


