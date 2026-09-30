#!/usr/bin/env python3
#
# Author: Pavlos Sermpezis (https://sites.google.com/site/pavlossermpezis/)
#

import os
import subprocess
import time
import numpy as np

PY3_BIN			= 'python3'
PY_SCRIPT 		= 'fair_scipy.py'
MAX_PROCESSES	= 1

# SCENARIOS_TO_TEST = [\
# 					([0.27, 0.63, 0.21, 0.37], './sim_results/sim_results_Ulastfm_pop1_a0.8_N2_C10_CPtop_Q0.5_L40.json'),\
# 					([0.29, 0.76, 0.22, 0.24], './sim_results/sim_results_Ulastfm_pop0_a0.99_N2_C10_CPtop_Q0.8_L40.json'),\
# 					([0.13, 0.6, 0.1, 0.4], './sim_results/sim_results_Ulastfm_pop0_a0.8_N5_C20_CPtop_Q0.5_L40.json'),\
# 					([0.15, 0.55, 0.09, 0.45], './sim_results/sim_results_Ulastfm_pop0_a0.99_N5_C10_CPtop_Q0.8_L40.json')\
# 					]

# SCENARIOS_TO_TEST = [\
# 					([0.27, 0.63, 0.2, 0.37], './sim_results/sim_results_Umovielens1k_pop1_a0.8_N2_C10_CPtop_Q0.5_L40.json'),\
# 					([0.29, 0.78, 0.22, 0.22], './sim_results/sim_results_Umovielens1k_pop1_a0.99_N2_C10_CPtop_Q0.8_L40.json'),\
# 					([0.16, 0.86, 0.12, 0.14], './sim_results/sim_results_Umovielens1k_pop0_a0.99_N5_C20_CPtop_Q0.5_L40.json'),\
# 					([0.15, 0.84, 0.11, 0.16], './sim_results/sim_results_Umovielens1k_pop0_a0.99_N5_C20_CPtop_Q0.8_L40.json')\
# 					]


# F = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]



'''
NEW SCENARIOS
'''
SCENARIOS_TO_TEST = [\
					### LastFM
					(['0.28', '0.58', '0.52', '0.17', '0.48'], './sim_results/sim_results_Ulastfm_pop0_a0.99_N2_C5_CPtop_Q0.9_L40.json'),\
					# (['0.87', '0.58', '0.52', '0.35', '0.35'], 0)
					# (['0.58', '0.58', '0.52', 'nan', 'nan'], 0)
					(['0.21', '0.71', '0.61', '0.15', '0.39'], './sim_results/sim_results_Ulastfm_pop0_a0.99_N2_C10_CPtop_Q0.9_L40.json'),\
					# (['0.85', '0.71', '0.61', '0.24', '0.24'], 0)
					# (['0.55', '0.71', '0.61', 'nan', 'nan'], 0)
					### Movielens
					# (['0.20', '0.51', '0.48', '0.10', '0.50'], './sim_results/sim_results_Umovielens1k_pop0_a0.8_N2_C5_CPtop_Q0.8_L40.json'),\
					# # (['0.60', '0.51', '0.48', '0.12', '0.12'], 0)
					# # (['0.24', '0.51', '0.48', 'nan', 'nan'], 0)
					(['0.27', '0.84', '0.65', '0.20', '0.35'], './sim_results/sim_results_Umovielens1k_pop1_a0.99_N2_C10_CPtop_Q0.9_L40.json'),\
					# (['0.82', '0.84', '0.65', '0.17', '0.17'], 0)
					# (['0.60', '0.84', '0.65', 'nan', 'nan'], 0)
					(['0.29', '0.80', '0.68', '0.15', '0.32'], './sim_results/sim_results_Umovielens1k_pop1_a0.99_N2_C5_CPtop_Q0.9_L40.json')\
					# (['0.87', '0.80', '0.68', '0.19', '0.19'], 0)
					# (['0.69', '0.80', '0.68', 'nan', 'nan'], 0)
					]



'''
NEW SCENARIOS - only TV
'''
# SCENARIOS_TO_TEST = [\
# 					### LastFM
# 					(['0.69', '0.34', '0.29', '0.40', '0.40'], './sim_results/sim_results_Ulastfm_pop0_a0.99_N5_C5_CPtop_Q0.9_L40.json'),\
# 					(['0.87', '0.58', '0.52', '0.35', '0.35'], './sim_results/sim_results_Ulastfm_pop0_a0.99_N2_C5_CPtop_Q0.9_L40.json'),\
# 					(['0.90', '0.70', '0.64', '0.26', '0.26'], './sim_results/sim_results_Ulastfm_pop0_a0.99_N2_C5_CPtop_Q0.8_L40.json'),\
# 					### Movielens
# 					(['0.86', '0.45', '0.39', '0.46', '0.46'], './sim_results/sim_results_Umovielens1k_pop0_a0.99_N10_C5_CPtop_Q0.5_L40.json'),\
# 					(['0.83', '0.48', '0.44', '0.39', '0.39'], './sim_results/sim_results_Umovielens1k_pop0_a0.99_N5_C5_CPtop_Q0.9_L40.json'),\
# 					(['0.87', '0.80', '0.68', '0.19', '0.19'], './sim_results/sim_results_Umovielens1k_pop1_a0.99_N2_C5_CPtop_Q0.9_L40.json')\
# 					]






F = dict()
F['max'] = [0.25, 0.20, 0.15, 0.10, 0.05, 0.01]
F['TV']  = [0.80, 0.60, 0.40, 0.20, 0.10, 0.01]
F['KL']  = [0.40]# [round(np.log(1.0/0.01)*i,2) for i in [0.60, 0.45, 0.30, 0.15, 0.05, 0.01]]

F_MODES = ['max', 'TV', 'KL']




FNULL = open(os.devnull, 'w')
active_processes = []

for max_fairness_mode in F_MODES:
	for sc in SCENARIOS_TO_TEST:
		U = sc[1].split('/sim_results_')[1].split('_')[0][1:] + '_U_matrix.mat'
		pop = sc[1].split('/sim_results_')[1].split('_')[1][3]
		a = sc[1].split('/sim_results_')[1].split('_')[2][1:]
		N = sc[1].split('/sim_results_')[1].split('_')[3][1:]
		C = sc[1].split('/sim_results_')[1].split('_')[4][1:]
		cp = sc[1].split('/sim_results_')[1].split('_')[5][2:]
		q = sc[1].split('/sim_results_')[1].split('_')[6][1:]
		L = sc[1].split('/sim_results_')[1].split('_')[7][1:].split('.json')[0]
		for f in F[max_fairness_mode]:
			qoe_eps = 0.1
			qoe_tol = 0.2
			# qoe_eps, qoe_tol = 1, 1 # Sanity check (deactivate QoE constraint)

			out_file = (
				'./sim_results/'
				'sim_results_U{}_pop{}_a{}_N{}_C{}_CP{}_Q{}_L{}_'
				'F{}{}_qoeE{}_qoeT{}.json'
			).format(
				U.split('_')[0], pop, a, N, C, cp, q, L,
				max_fairness_mode, f,
				qoe_eps, qoe_tol
			)
			if os.path.exists(out_file):
				print('This file already exists: {}'.format(out_file))
				continue
			cmd = '{} {} -U_related {} -popularity {} -clickthrough {} -batch_size {} -cache_size {} -cache {} -qref {} -session_length {} -o {} -fairness_mode {} -fair_weight {} -qoe_tol {} -qoe_eps {}'.format(PY3_BIN, PY_SCRIPT, U, pop, a, N, C, cp, q, L, out_file, max_fairness_mode, f, qoe_tol, qoe_eps)
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
