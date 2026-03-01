import re
import os
import numpy as np

def kl(p,q,alpha=0):
	'''
	Returns the KL distance between the two distributions p1 and p2. In case the parameter "alpha" is >0, it returns a smoothed version of the KL distance [Steck18].
	If the KL distance diverges (becomes inf) then the function returns the value np.nan

	[Steck18] Harald Steck. 2018. Calibrated recommendations. In ACM RecSys. ACM, 154–162
	'''
	d = [np.nan]*len(p) # initialize the list with the kl distance per item in the distribution
	q_smooth = [(1-alpha)*q[i]+ alpha*p[i]  for i in range(len(p))]
	for i in range(len(p)):
		if p[i]==0:
			d[i] = [0]
		else:
			if q_smooth[i]==0:
				d[i] = np.nan # the KL metrics diverges (becomes inf)
			else:
				d[i] = p[i]*np.log(p[i]/q_smooth[i])
	if alpha==0:
		return np.sum(d)
	else:
		return np.sum(d) / np.log(1/alpha)
    
def p_fairness(pNA,pBS,metric):
	'''
	This method receives two lists, representing content demand distributions (p[i] is the fraction of demand for the i^{th} content), 
	and a string that denotes which fairness measure will be used, 
	and returns a value for the fairness measure (or, "difference") of the given lists
	Input:
		pNA: a list of the first content demand distribution; this correspongs to the network-aware (NA) RS case
		pBS: a list of the second content demand distribution (should be of the same length with pNA); this correspongs to the baseline (BS) RS case
		metric: a string to denote which fairness measure to be used; it can take values {'avg', 'sum', 'max', 'kl', 'kl-smooth'}

	'''
	if metric=='avg':
		f = 0.5 * np.sum([np.abs(pNA[i]-pBS[i]) for i in range(len(pNA))])
	elif metric=='sum':
		f = np.sum([np.abs(pNA[i]-pBS[i]) for i in range(len(pNA))])
	elif metric=='max':
		r = [np.abs(pNA[i]-pBS[i]) for i in range(len(pNA))]
		f = np.max([np.abs(pNA[i]-pBS[i]) for i in range(len(pNA))])
	elif metric=='kl':
		f = kl(pNA,pBS)
	elif metric=='kl-inv':
		f = kl(pBS,pNA)
	elif metric=='kl-smooth':
		f = kl(pNA,pBS,alpha=0.01)
	elif metric=='kl-smooth-inv':
		f = kl(pBS,pNA,alpha=0.01)
	else:
		raise ValueError('The given metric "{}" is not within the available options'.format(metric))
	return f

def match_regex_array(files, regex):
    matches = [re.findall(regex, f) for f in files]
    matches = ["./sim_results/" + m[0] for m in matches if len(m) > 0 ]
    return matches

def get_selected_scenarios(simulator_path):    
    U_MATRIX        = ['lastfm_U_matrix.mat', 'movielens1k_U_matrix.mat']
    POPULARITY      = [0, 1]
    ALPHA           = [0.99, 0.8, 0.5]
    NN              = [2, 5, 10]
    CACHE           = [10]#[5, 10, 20]
    POLICY          = ['top']
    QQ              = [0.5, 0.8, 0.9, 1.0]
    LL              = [-1, 1, 40]   # L=1 corresponds to single-step (or, myopic) NA optim algorithm, and L=40 to sequential NA optim algorithm
    W_bfs           = [1, 2] # this is times the N, i.e., 1*N, 2*N
    D_bfs           = [2]
    files = []
    i=1
    for U in U_MATRIX:
        if U == 'lastfm_U_matrix.mat':
            K = 757
        else:
            K = 1060
        for pop in POPULARITY:
            for a in ALPHA:
                for N in NN:
                    for C in CACHE:
                        for cp in POLICY:
                            for q in QQ:
                                for L in LL:
                                    for W in W_bfs:
                                        for D in D_bfs:
                                            if L>0:
                                                out_file = simulator_path + 'sim_results/sim_results_U{}_pop{}_a{}_N{}_C{}_CP{}_Q{}_L{}.json'.format(U.split('_')[0], pop, a, N, C, cp, q, L)
                                            else: 
                                                out_file = simulator_path + 'sim_results/sim_results_U{}_pop{}_a{}_N{}_C{}_CP{}_Q{}_cabaret_W{}_D{}.json'.format(U.split('_')[0], pop, a, N, C, cp, q, W, D)
                                            if not os.path.exists(out_file):
                                                continue
                                            if L==1:
                                                plot_format = 'ok'
                                            elif L==40:
                                                plot_format = 'xr'
                                            elif L==-1:
                                                plot_format = '^b'
                                            else:
                                                error('Not correct parameter L')
                                            try:
                                                files.append(out_file.split("/")[-1])
                                            except Exception as e:
                                                print(e)
                                                continue
    return files