import scipy.io as sio
import numpy as np
import copy
import random
import math
import timeit
import argparse
from numpy import linalg as LA
from itertools import combinations
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
import multiprocessing
import time
import json
from json import JSONEncoder
import hashlib


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def lp_solver(item_cost, cache, N, q_percentage, qmax_vector, p0, pBS, alpha, u, fairness_mode, weight,
              *, kl_cut_max=8.0, kl_cut_step=0.05, method="highs"):
    """Solve the Fair-NFR LP with scipy.optimize.linprog.

    Variable layout:
      p_i              -> indices [0, K)
      w_ij             -> indices [K, K + K*K), row-major
      z_i (TV/KL only) -> indices [K + K*K, K + K*K + K)

    Returns:
      policy R, where R[i, j] = w_ij / p_i.
    """
    p0 = np.asarray(p0, dtype=float)
    pBS = np.asarray(pBS, dtype=float)
    u = np.asarray(u, dtype=float)
    qmax_vector = np.asarray(qmax_vector, dtype=float)
    cache = set(cache)
    lib_size = len(p0)

    n_p = lib_size
    n_w = lib_size ** 2
    has_z = fairness_mode in {"TV", "KL"}
    n_z = lib_size if has_z else 0
    n_vars = n_p + n_w + n_z

    def p_idx(i):
        return i

    def w_idx(i, j):
        return lib_size + i * lib_size + j

    def z_idx(i):
        return lib_size + n_w + i

    # CPLEX code minimized delivery cost: cached items cost 0, uncached item_cost.
    # This is equivalent to maximizing cache-hit probability because sum_i p_i = 1.
    objective = np.zeros(n_vars, dtype=float)
    for i in range(lib_size):
        objective[p_idx(i)] = 0.0 if i in cache else float(item_cost)

    bounds = [(0.0, None)] * n_vars
    if fairness_mode == "KL":
        for i in range(lib_size):
            bounds[z_idx(i)] = (None, 0.0)

    eq_rows, eq_cols, eq_data, b_eq = [], [], [], []
    ub_rows, ub_cols, ub_data, b_ub = [], [], [], []

    def add_eq(cols, vals, rhs):
        row = len(b_eq)
        eq_rows.extend([row] * len(cols))
        eq_cols.extend(cols)
        eq_data.extend(vals)
        b_eq.append(float(rhs))

    def add_ub(cols, vals, rhs):
        row = len(b_ub)
        ub_rows.extend([row] * len(cols))
        ub_cols.extend(cols)
        ub_data.extend(vals)
        b_ub.append(float(rhs))

    # sum_i p_i = 1
    add_eq([p_idx(i) for i in range(lib_size)], [1.0] * lib_size, 1.0)

    # Budget constraints: sum_j w_ij - N*p_i = 0
    for i in range(lib_size):
        add_eq(
            [w_idx(i, j) for j in range(lib_size)] + [p_idx(i)],
            [1.0] * lib_size + [-float(N)],
            0.0,
        )

    # QoR constraints: sum_j w_ij*u_ij - p_i*q*qmax_i >= 0
    # linprog uses <=, therefore: -sum_j w_ij*u_ij + p_i*q*qmax_i <= 0
    for i in range(lib_size):
        add_ub(
            [w_idx(i, j) for j in range(lib_size)] + [p_idx(i)],
            [-float(u[i, j]) for j in range(lib_size)] + [float(q_percentage * qmax_vector[i])],
            0.0,
        )

    # Stationarity: p_j - (alpha/N)*sum_i w_ij = (1-alpha)*p0_j
    for j in range(lib_size):
        add_eq(
            [w_idx(i, j) for i in range(lib_size)] + [p_idx(j)],
            [-float(alpha) / float(N)] * lib_size + [1.0],
            (1.0 - alpha) * p0[j],
        )

    # No self recommendations: w_ii = 0
    for i in range(lib_size):
        add_eq([w_idx(i, i)], [1.0], 0.0)

    # Box constraints from r_ij <= 1 and w_ij = p_i*r_ij: w_ij - p_i <= 0
    for i in range(lib_size):
        pi = p_idx(i)
        for j in range(lib_size):
            add_ub([w_idx(i, j), pi], [1.0, -1.0], 0.0)

    # Fairness constraints
    if fairness_mode in {"max", "perMax"}:
        eps = 1e-15
        for i in range(lib_size):
            if fairness_mode == "max":
                add_ub([p_idx(i)], [1.0], pBS[i] + weight)
                add_ub([p_idx(i)], [-1.0], -pBS[i] + weight)
            else:
                denom = max(float(pBS[i]), eps)
                add_ub([p_idx(i)], [1.0 / denom], 1.0 + weight)
                add_ub([p_idx(i)], [-1.0 / denom], -1.0 + weight)

    elif fairness_mode == "TV":
        # F_tv = 0.5 * sum_i |p_i - pBS_i| <= weight
        add_ub([z_idx(i) for i in range(lib_size)], [1.0] * lib_size, 2.0 * weight)
        for i in range(lib_size):
            # z_i >= pBS_i - p_i  -> -p_i - z_i <= -pBS_i
            add_ub([p_idx(i), z_idx(i)], [-1.0, -1.0], -pBS[i])
            # z_i >= p_i - pBS_i  ->  p_i - z_i <=  pBS_i
            add_ub([p_idx(i), z_idx(i)], [1.0, -1.0], pBS[i])

    elif fairness_mode == "KL":
        # Same linear-cut approximation as the original CPLEX code:
        #   z_i <= tangent approximation of log(p_i)
        #   sum_i pBS_i*z_i >= sum_i pBS_i*log(pBS_i) - weight*log(100)
        # The log(100) factor matches the original normalization with w=0.01.
        eps = 1e-300
        pbs_safe = np.maximum(pBS, eps)
        add_ub(
            [z_idx(i) for i in range(lib_size)],
            [-float(pBS[i]) for i in range(lib_size)],
            -float(np.sum(pBS * np.log(pbs_safe))) + float(weight) * np.log(1.0 / 0.01),
        )

        for num in np.arange(0.0, kl_cut_max, kl_cut_step):
            exp_num = float(np.exp(num))
            rhs = -1.0 - float(num)
            for i in range(lib_size):
                # z_i - exp(num)*p_i <= -1 - num
                add_ub([p_idx(i), z_idx(i)], [-exp_num, 1.0], rhs)

    elif fairness_mode is not None:
        raise ValueError("Unknown fairness_mode={!r}".format(fairness_mode))

    A_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(len(b_eq), n_vars)).tocsr()
    A_ub = coo_matrix((ub_data, (ub_rows, ub_cols)), shape=(len(b_ub), n_vars)).tocsr()

    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=np.asarray(b_ub, dtype=float),
        A_eq=A_eq,
        b_eq=np.asarray(b_eq, dtype=float),
        bounds=bounds,
        method="highs-ds",
    )

    if not result.success:
        raise RuntimeError("linprog failed: status={}, message={}".format(result.status, result.message))

    sol_final = np.asarray(result.x)
    pi = sol_final[:lib_size]
    sol_matrix = sol_final[lib_size: lib_size + n_w]
    W = np.reshape(sol_matrix, (lib_size, lib_size))

    policy = np.divide(W, pi[:, None], out=np.zeros_like(W), where=pi[:, None] > 1e-14)
    return policy


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

def isclose(a, b):
    rel_tol=1e-5
    abs_tol=1e-5
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def recomList(N, r_nz, x, item):
    """
    Input: N (num of recs, an integer), r_nz
    """
    list_ind = [[] for i in range(N)]
    list_pmf = [[] for i in range(N)]
    recommended = []
    j = 0
    s = 0
    for i in range(N):
        while (s < 1.0) and isclose(s, 1.0) == False:
            list_pmf[i].append(r_nz[j][1])
            list_ind[i].append(r_nz[j][0])
            s = sum(list_pmf[i])
            j += 1
        s = s - 1.0
        if (s > 0) and isclose(s, 0.0) == False:
            list_pmf[i][-1] = list_pmf[i][-1] - s
            list_pmf[i+1].append(s)
            list_ind[i+1].append(r_nz[j-1][0])
            s = 0

    for i in range(N):
        np.random.seed(x)
        list_pmf[i] = projection_simplex_sort(np.asarray(list_pmf[i]), z=1)
        recommended.append(np.asscalar(np.random.choice(list_ind[i], 1, p = list_pmf[i])))
    return recommended

def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)

def recomm_dictionary(r):
    dict = {}
    keys = range(r.shape[1])
    for i in keys:
        temp = r[i,:]
        dict[i] = [[j,temp[j]] for j, val in enumerate(temp) if val != 0]
    return dict

def measure_clickthrough(lam, R, u, alpha, qmax_vector, N, p0, runs, model, item_selection):
    k = len(p0)
    dict = recomm_dictionary(R)
    content_lib = range(k)
    user_satisfaction_vec = []
    recommendation_clicks_vec = []
    happiness_vec = []

    for t in range(runs):

        # First request through the search bar and check if hit or not
        content = np.asscalar(np.random.choice(content_lib, 1, p = p0))
        num_requests = np.asscalar(np.random.geometric(p = 1 - lam, size = 1))
        # num_requests = time

        user_satisfaction_accum = 0
        recommendation_clicks = 0
        utility = 0
        user_satisfaction_time_average = []
        clickthrough_time_average = []

        # Go through the requests
        for l in range(num_requests):

            tem = content
            x = random.randint(1,1000)
            recom_nz = dict[int(content)]
            list_rec_items = recomList(N, recom_nz, x, int(content))
            u_list = np.array([u[content,j] for j in list_rec_items])
            quality_instant = 1.0*np.sum(u_list)/qmax_vector[content] # user statisfaction instant

            if model == "1":
                alpha_sim = alpha
            if model == "2":
                alpha_sim = quality_instant

            user_satisfaction_accum += quality_instant

            # Update
            user_satisfaction_time_average.append(1.0*user_satisfaction_accum/(l+1))

            if np.random.rand() <= alpha_sim:
                recommendation_clicks += 1
                if item_selection == "prop":
                    pmf = projection_simplex_sort(1.0*u_list/(np.sum(u_list)), z=1)
                    content = np.asscalar(np.random.choice(list_rec_items, 1, p = pmf))
                if item_selection == "unif":
                    pmf = (1.0/N)*np.ones([len(u_list)])
                    content = np.asscalar(np.random.choice(list_rec_items, 1, p = pmf))
                utility += u[tem,content]
            else:
                content = np.asscalar(np.random.choice(content_lib, 1, p = p0))

            # Update
            # clickthrough_time_average.append(1.0*recommendation_clicks/(l+1))
            # clickthrough_time_average.append(1.0*recommendation_clicks/(l+1))

        user_satisfaction_vec.append(1.0*user_satisfaction_accum/num_requests)

        if recommendation_clicks > 0:
            happiness_vec.append(1.0*utility/recommendation_clicks)

        recommendation_clicks_vec.append((1.0*recommendation_clicks)/(num_requests))

    return np.mean(user_satisfaction_vec), np.mean(recommendation_clicks_vec), np.mean(happiness_vec), user_satisfaction_time_average, clickthrough_time_average

def mdp_inner_minimizer_cplex_user1(i, K, N, value, q_percentage, qmax_vector, u_i):
    u_i = u_i.tolist()

    # Create an instance of a linear problem to solve
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    objective = value

    lower_bounds = [0.0]*K
    upper_bounds = [1]*K
    problem.variables.add(obj = objective, lb = lower_bounds, ub = upper_bounds)

    idxs = range(K)
    con1 = [idxs,[1]*K]
    senses1 = ['E']*1

    con2 = []
    q = qmax_vector[i]*q_percentage
    l1 = [-item for item in u_i]
    con2 = [idxs,l1]
    senses2 = ['L']*1

    l1 = [0]*K
    l1[i] = 1
    con3 = [idxs,l1]
    senses3 = ['E']*1

    my_senses = senses1 + senses2 + senses3
    my_rhs = [0]*3
    my_rhs[0] = N
    my_rhs[1] = -1.0*q
    my_rhs[2] = 0

    constraints = [con1,con2,con3]
    problem.linear_constraints.add(lin_expr = constraints, senses = my_senses, rhs = my_rhs)
    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    problem.set_results_stream(None)

    # Solve the problem
    problem.solve()
    sol_vec = problem.solution.get_values()
    sol_final = np.asarray(sol_vec)
    return sol_final

def cache_hit_ratio(x, p0, N, R, alpha):
    n = len(x)
    g = np.identity(n)-1.0*(alpha/N)*R
    f1 = (1-alpha)*np.dot(p0.T,np.linalg.inv(g))
    f2 = np.dot(f1,x)
    return f2

def lcr_indexes(u_i, cost_vec, ind):
    """
    Inputs:
    u_i: i-th row of content relations - array size kx1
    cost_vec: cost of contents - array size kx1
    ind: sorted indices of the contents cost - array of size kx1
    out: indixes array with RELATED that are also sorted by increasing cost - array of size as many as the related files for content i
    """
    u = u_i.tolist()
    s = cost_vec.tolist()
    lcr_ind = [ind[i] for i in range(len(s)) if u[ind[i]]>0.0]
    return np.array(lcr_ind)

def solve_lin_prog_fast_binaryU_case(cost_vec, u_i, N, qref, qmax_vector, i):

    """
    Inputs:
    cost_vec: cost of contents - array size kx1
    u_i: i-th row of content relations - array size kx1
    N: nb of recoms - integer
    q: fraction of maximum quality expressed in a decimal - a floating point number
    i: which content are optimizing for - integer
    Output: the policy for content i - an array that sums to N
    """
    K = len(cost_vec)
    r = np.zeros([K])

    u_i[i] = 0.0 # zero i,i entry
    temp = cost_vec[i]
    cost_vec[i] = 1e10
    ind = np.argsort(cost_vec)
    lcr_ind = lcr_indexes(u_i, cost_vec, ind)

    if (qmax_vector[i] >= N):
        qm = N
    else:
        qm = qmax_vector[i]

    q_required = qref*qm
    q_current = 0
    budget_rem = N
    counter = 0

    while (q_required > q_current):
        if (q_required - q_current <= 1.0):
            r[lcr_ind[counter]] = q_required - q_current
            exception = lcr_ind[counter]
            budget_rem = budget_rem - (q_required - q_current)
            budget_left_for_exception = 1.0 - (q_required - q_current)
            break
        r[lcr_ind[counter]] = 1.0
        q_current = q_current + 1.0
        budget_rem = budget_rem - 1.0
        ind = ind[(ind != lcr_ind[counter])]
        counter = counter + 1

    # Quality is filled, now assign values to least cost remaining items
    counter = 0
    while (budget_rem > 0):
        if ind[counter] == exception:
            r[exception] = budget_left_for_exception + r[exception]
            counter = counter + 1
            budget_rem = budget_rem - budget_left_for_exception
        else:
            if (budget_rem >= 1.0):
                r[ind[counter]] = 1.0
                counter = counter + 1
                budget_rem = budget_rem - 1.0
            else:
                r[ind[counter]] = budget_rem
                break
    cost_vec[i] = temp

    return r

def zipf_pmf(lib_size, expn):
    K = lib_size
    p0 = np.zeros([K])
    for i in range(K):
        p0[i] = ((i+1)**(-expn))/np.sum(np.power(range(1,K+1), -expn))
    return p0


def cabaret(u, N, W_bfs, D_bfs, cache_set):
    top_related_per_content = dict()

    K = len(u)
    R = np.zeros([K,K])
    for i in range(K):
        top_related_per_content[i] = np.argsort(u[i,:])[-W_bfs:][::-1]

    for i in range(K):
        nodes_to_check = []
        checked_nodes = []
        L = []
        nodes_to_check.extend( top_related_per_content[i] )
        checked_nodes.append(i)
        L.extend( top_related_per_content[i] )
        current_D_bfs = D_bfs - 1
        while current_D_bfs >0:
            current_nodes = []
            for j in nodes_to_check:
                checked_nodes.append(j)
                current_nodes.extend( top_related_per_content[j] )
            nodes_to_check = [n for n in current_nodes if n not in checked_nodes]
            for j in current_nodes:
                if (j not in L) and (j!=i):
                    L.append(j)
            current_D_bfs = current_D_bfs - 1

        cabaret_recommendations = [n for n in L if n in cache_set]
        cabaret_recommendations.extend([n for n in L if n not in cache_set])
        cabaret_recommendations = cabaret_recommendations[0:N]

        for j in cabaret_recommendations:
            R[i,j] = 1.0

    return R


def _policy_digest(R, decimals=10):
    Rq = np.round(np.asarray(R, dtype=np.float64), decimals=decimals)
    return hashlib.blake2b(Rq.tobytes(), digest_size=16).hexdigest()


def compute_policy(
    cost_UNcached, lam, u, N, p0, cache, qmax_vector, qref, alpha, case,
    max_iter=500,
    verbose=True,
    detect_cycles=True,
    return_best_on_cycle=True,
    tie_break_eps=1e-10,
):
    """
    Safer version of the old policy iteration.

    Why this exists:
      - The original loop has no max_iter.
      - With high alpha/lambda and degenerate inner LPs, policy improvement can cycle.
      - Returning the best policy seen is safer than hanging forever.
    """
    K = len(p0)
    cost_cached = 0.0

    x = cost_UNcached * np.ones(K)
    x[cache] = cost_cached

    R = topN(N, u)
    R_old = np.ones((K, K))
    value_old = np.ones(K)

    def score_policy(R_candidate):
        # Expected uncached cost under stationary demand. Smaller is better.
        try:
            pi = compute_pi(R_candidate, p0, alpha, N)
            return float(np.dot(pi, x))
        except Exception:
            return float("inf")

    best_R = copy.deepcopy(R)
    best_score = score_policy(R)
    seen = {}

    if case == "model1":
        eps1 = 0.01
        eps2 = 0.001

        for it in range(1, max_iter + 1):
            value = policy_evaluation(
                lam, alpha, N, cache, cost_UNcached, p0, u, qmax_vector, R, "model1"
            )

            num_cores = multiprocessing.cpu_count()
            res_parallel = Parallel(n_jobs=num_cores)(
                delayed(mdp_inner_minimizer_scipy_user1)(
                    k, K, N, value, qref, qmax_vector, u[k, :],
                    previous_r=R[k, :],
                    tie_break_eps=tie_break_eps,
                )
                for k in range(K)
            )
            R_new = np.vstack(res_parallel)

            diff1 = float(np.linalg.norm(R_new - R_old))
            diff2 = float(np.linalg.norm(value - value_old, 2))
            current_score = score_policy(R_new)

            if current_score < best_score:
                best_score = current_score
                best_R = copy.deepcopy(R_new)

            if verbose and (it == 1 or it % 10 == 0):
                print(
                    f"compute_policy iter={it} "
                    f"diff_R={diff1:.6g} diff_value={diff2:.6g} "
                    f"uncached_cost={current_score:.10g} best={best_score:.10g}",
                    flush=True,
                )

            if (diff1 < eps1) or (diff2 < eps2):
                if verbose:
                    print(
                        f"compute_policy converged at iter={it}: "
                        f"diff_R={diff1:.6g}, diff_value={diff2:.6g}",
                        flush=True,
                    )
                return R_new

            if detect_cycles:
                key = _policy_digest(R_new, decimals=10)
                if key in seen:
                    msg = (
                        f"compute_policy cycle detected: iter {seen[key]} -> {it}; "
                        f"returning best policy seen with uncached_cost={best_score:.10g}"
                    )
                    if return_best_on_cycle:
                        print(msg, flush=True)
                        return best_R
                    raise RuntimeError(msg)
                seen[key] = it

            R_old = copy.deepcopy(R)
            R = R_new
            value_old = copy.deepcopy(value)

        msg = (
            f"compute_policy did not converge after {max_iter} iterations; "
            f"returning best policy seen with uncached_cost={best_score:.10g}"
        )
        if return_best_on_cycle:
            print(msg, flush=True)
            return best_R
        raise RuntimeError(msg)

    raise NotImplementedError("This patch only replaces the model1 branch.")

def policy_evaluation(lam, alpha, N, cache, cost_UNcached, p0, u, qmax_vector, R, case):

    epsil = 0.0001 # stopping criterion for policy evaluation

    K = len(p0)
    cost_cached = 0.0
    x = cost_UNcached*np.ones(K)
    x[cache] = cost_cached
    v = np.ones(K)*1

    while True:
        delta = 0
        for k in range(K):
            temp = v[k]
            if (case == "model1"):
                v[k] = x[k] + lam*(1-alpha)*np.dot(p0,v) + lam*alpha*np.sum(np.dot((1.0/N)*R[k,:],v))
            if (case == "model2"):
                c = np.multiply(u[k,:], v) - np.dot(p0,v)*u[k,:]
                v[k] = x[k] + lam*(1.0/qmax_vector[k])*np.dot(c,R[k,:]) + lam*np.dot(p0,v)
            delta = max([delta,abs(temp-v[k])])
        if delta < epsil:
            break

    return v

def mdp_inner_minimizer_sorting_user2(u_i, v, p0, N, R_i, c, i):
    K = len(v)
    t = c[i]
    c[i] = np.PINF
    indexes = sorted(range(len(c)), key = lambda k: c[k])[:N]
    R_i = np.zeros(K)
    R_i[indexes] = 1.0
    c[i] = t
    return R_i

def synthetic_U(lib_size, case):
    val = 0.5 # lower level of utility you can get

    u = np.zeros([lib_size,lib_size])
    for i in range(lib_size):

        contents = list(range(lib_size))
        contents.remove(i) # you need that so you u[i,i] is always zero

        upper = np.ceil(0.02*lib_size)
        upper = 50
        how_many_related = random.randint(1, upper) # you have N + a random number between 0 and 2
        y = np.random.choice(contents, how_many_related, replace = False)

        if (case == "binary"):
            u[i,y] = 1
        if (case == "continuous"):
            u[i,y] = np.random.uniform(val, 1, len(y))
    return u

def compute_qmax(u, N):
    # returns the qmax vector which shows the max quality you can get for each content
    k = len(u)
    qmax = [sum(sorted(u[i,:])[-N:]) for i in range(k)]
    return qmax

def hit(request, cache):
    """
    Input: cache (a list), request (integer)
    Out: 1 or 0 (integer)
    """
    h = request in cache
    return int(h == True)

def mdp_inner_minimizer_scipy_user1(
    i,
    K,
    N,
    value,
    q_percentage,
    qmax_vector,
    u_i,
    previous_r=None,
    tie_break_eps=1e-8,
):
    
    A_eq = np.zeros((2, K))
    A_eq[0, :] = 1.0
    A_eq[1, i] = 1.0
    b_eq = np.array([N, 0.0])

    A_ub = -u_i.reshape(1, K)
    b_ub = np.array([-q_percentage * qmax_vector[i]])

    bounds = [(0.0, 1.0)] * K

    c = np.asarray(value, dtype=float).copy()

    # Tiny tie-break: among near-equivalent optima, prefer the previous row.
    if previous_r is not None:
        previous_r = np.asarray(previous_r, dtype=float)
        c = c + tie_break_eps * (1.0 - previous_r)

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ds",
    )

    if not res.success:
        raise RuntimeError(f"Inner LP failed for row {i}: {res.message}")

    return res.x



def get_alpha_vector(R, u, qmax_vector):
    res_qual = np.sum(np.multiply(R,u), axis = 1)
    return [1.0*res_qual[i]/qmax_vector[i] for i in range(len(u))]

def topN(N, u):
    K = len(u)
    R = np.zeros([K,K])
    for i in range(K):
        sss = u[i,:].argsort()[-N:][::-1]
        ind = list(sss)
        R[i,ind] = 1.0
    return R

def LowC(N, cache, u):
    k = len(u)
    r = np.zeros([k,k])
    for i in range(k):
        temp = np.argsort(-1.0*u[i,:])
        temp = temp[temp != i]
        temp1 = [el for el in temp if el in cache]
        temp2 = [el for el in temp if el not in cache]
        it1 = 0
        it2 = 0
        for l in range(N):
            if (l < len(temp1)):
                r[i,temp1[it1]] = 1.0
                it1 += 1
            else:
                r[i,temp2[it2]] = 1.0
                it2 += 1

    return r

def compute_pi(R, p0, alpha, N):
    n = len(p0)
    g = np.identity(n) - 1.0*(alpha/N)*R
    return (1-alpha)*np.dot(p0.T, np.linalg.inv(g))

def cache_policy(option, cache_size, p_bs):
    # this will return a list with len = cache_size
    if option == "top":
        idx_list = list(np.argsort(p_bs))
        return idx_list[-cache_size:]
    if option == "random":
        return np.random.choice(range(len(p_bs)), cache_size, replace = False)

def zero_diagonal_u(u):
    for i in range(len(u)):
        u[i,i] = 0

def main():

    # to test it run: python fair.py -U_related U_matrix.mat -popularity 0.2 -clickthrough 0.7 -batch_size 3 -cache_size 4 -cache "top" q_ref 0.7 -session_length 20

    # parsing
    parser = argparse.ArgumentParser(description="Returns policy and pi_final")
    parser.add_argument('-U_related', '--U_matrix', dest='U_matrix', type=str, required=True)
    parser.add_argument('-popularity', '--z_param', dest='z_param', type=float, required=True)
    parser.add_argument('-clickthrough', '--alpha', dest='alpha', type=float, required=True)
    parser.add_argument('-batch_size', '--N', dest='N', type=int, required=True)
    parser.add_argument('-cache_size', '--cs', dest='cs', type=int, required=True)
    parser.add_argument('-cache', '--policy', dest='policy', type=str, required=True)
    parser.add_argument('-qref', '--q', dest='q', type=float, required=True)
    parser.add_argument('-session_length', '--L', dest='L', type=int, required=True)
    parser.add_argument('-o', '--out_file', dest='out_file', type=str, required=True)
    # new added arguments
    parser.add_argument('-fairness_mode', '--fairness_mode', dest='fairness_mode', type=str)
    parser.add_argument('-fair_weight', '--weight', dest='weight', type=float)
    parser.add_argument('-cab', '--cabaret_parameters', dest='cab', nargs=2, metavar=('W_bfs', 'D_bfs'), help='The values for the width and depth of the BFS in the CABaRet algorithm.', type=int)

    args = parser.parse_args()

    # load relation matrix
    u_data = sio.loadmat(args.U_matrix)
    u = u_data['u']
    zero_diagonal_u(u) # zero the diagonal in case there are nonzero elems
    K = len(u) # catalog size

    # create a p0 ~ zipfian
    p0 = zipf_pmf(K, args.z_param)

    # get alpha of user
    alpha = args.alpha

    # get alpha of user
    N = args.N

    # which max_fairness
    fairness_mode = args.fairness_mode

    # how much weight on the constraint, for w >= 1 the constraint is inactive
    weight = args.weight

    # find top-N policy (according to U best items)
    R_top = topN(N, u)

    # find pi_baseline
    pi_bs = compute_pi(R_top, p0, alpha, N)

    # Create cache set
    cache_size = args.cs
    cache_set = cache_policy(args.policy, cache_size, pi_bs)

    # Constraint of recommendation accuracy for the user
    qref = args.q

    # find q_max vector
    qmax_vector = compute_qmax(u, N)

    # Average user session size
    lam = 1 - 1.0/args.L

    # un-cached items cost 1.0, cached ones 0.0
    content_cost = 1.0

    if args.L ==-1: # L=-1 denotes that the NA algorithm is CABaRet
        W_bfs = args.cab[0]
        D_bfs = args.cab[1]
        R_NA = cabaret(u, N, W_bfs, D_bfs, cache_set)
    else:
        if fairness_mode is None:
            R_NA = compute_policy(content_cost, lam, u, N, p0, cache_set, qmax_vector, qref, alpha, "model1")
        else:
            R_NA = lp_solver(content_cost, cache_set, N, qref, qmax_vector, p0, pi_bs, alpha, u, fairness_mode, weight)

    pi_final_NA = compute_pi(R_NA, p0, alpha, N)

    # put in a dictionary the results of the NA RS
    outData = {"R_NA": R_NA, "R_top": R_top, "pi_final_NA": pi_final_NA, "pi_bs": pi_bs, "p0": p0}

    with open(args.out_file, "w") as write_file:
        json.dump(outData, write_file, cls=NumpyArrayEncoder)


if __name__ == "__main__":
    main()
