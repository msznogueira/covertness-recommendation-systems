import re
import os

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