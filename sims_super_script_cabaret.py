#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#

import os
import subprocess
import time


PY3_BIN			= 'python3'
PY_SCRIPT 		= 'fair_scipy.py'
MAX_PROCESSES	= 1

U_MATRIX		= ['lastfm_U_matrix.mat', 'movielens1k_U_matrix.mat']
POPULARITY		= [0, 1]
ALPHA			= [0.99, 0.8, 0.5]
NN 				= [2, 5, 10]
CACHE			= [5, 10, 20]
POLICY 			= ['top'] # are there any other policies implemented now in the simulator?
QQ 				= [1.0]
LL 				= [-1] #@THEO: I wanted to have the single-step policy and the wowmom policy. Fix the numbers accordingly.
W_bfs 			= [1, 2] # this is times the N, i.e., 1*N, 2*N
D_bfs 			= [1, 2]



FNULL = open(os.devnull, 'w')
active_processes = []

for U in U_MATRIX:
	for pop in POPULARITY:
		for a in ALPHA:
			for N in NN:
				for C in CACHE:
					for cp in POLICY:
						for q in QQ:
							for L in LL:
								for W in W_bfs:
									for D in D_bfs:
										out_file = './sim_results/sim_results_U{}_pop{}_a{}_N{}_C{}_CP{}_Q{}_cabaret_W{}_D{}.json'.format(U.split('_')[0], pop, a, N, C, cp, q, W, D)
										if os.path.exists(out_file):
											print('This file already exists: {}'.format(out_file))
											continue
										if (pop > 0) and (a > 0.95):
											continue # when alpha is too large, the distributions p0 does not affect much the results.
										if (W ==1) and (D==1):
											continue # this is the BS-RS
										cmd = '{} {} -U_related {} -popularity {} -clickthrough {} -batch_size {} -cache_size {} -cache {} -qref {} -session_length {} -o {} -cab {} {}'.format(PY3_BIN, PY_SCRIPT, U, pop, a, N, C, cp, q, L, out_file,W*N,D)
										while len(active_processes) >= MAX_PROCESSES:
											for p in active_processes:
												if p.poll() is not None:
													active_processes.remove(p)
													print('Time elapsed: {}'.format(time.time()-t0))
												break
											time.sleep(1)
										print(cmd)
										p = subprocess.Popen(cmd.split(' '))
										active_processes.append(p)
										t0 = time.time()
