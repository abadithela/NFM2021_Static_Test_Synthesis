#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:32:32 2020

@author: apurvabadithela
"""


## This file contains helper functions for running large examples of the restrict_transitions algorithm:

import numpy as np
import time
import ipdb
import random
from src.grid_functions import construct_grid
import networkx as nx
import gurobipy as gp
import scipy.sparse as sp
from gurobipy import GRB
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from src.aug_path import get_augmenting_path
from networkx.generators.random_graphs import fast_gnp_random_graph, gnm_random_graph
from networkx.algorithms.flow.utils import build_residual_network
from src.milp_functions import cut_aug_paths, milp, keep_aug_paths, construct_milp_params, find_nkeep, augment_paths, sdistance, construct_c
from src.restrict_transitions_cycles import remove_edges_corrected_cycle, remove_edges
from src.preprocess_constraints_SAP import SAP_sets
from src.milp_functions import get_SAP_constraints

# Functions for running repeated experiments on gridworlds:
# Running 10 iterations for every experiment configuration:
# Returns average time taken, no. of simulations that timed out,  
# and avg. total_no_iterations across runs that didn't fail or time out, and percentage of failed instances (due to algorithm not being complete)
def run_iterations(iM, iN, inP, Niter, allow_diag, KEY):
    time_avg = 0
    ntimed_out = 0
    total_iter_avg = 0 # Total no. of while loops running this algorithm
    nfail = 0
    ii = 0
    nstates = iM*iN
    total_ILP_avg = 0 # Total no. of ILP iterations for this algorithm
    itotal = 0
    TIME_AVG_ARR = []
    while ii < Niter and itotal<100:  # Parsing through iterations
        ns = inP + 1 # Total no. of samples: inP + goal
        G = construct_graph(iM, iN, allow_diag) # grids
        # G = construct_random_graph(n, m)
        props, skip_graph = generate_valid_config(G, ns, nstates)
        
        # Enter only valid propositions:
        if not skip_graph:
            t = time.time()
            print("Start run: "+str(ii))
            Q0 = props[0]
            C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main,skip_iter = remove_edges(G, props, Q0,KEY)
            if not skip_iter:
                elapsed_time = time.time() - t # Time that has elapsed
                
                
                if discard=="time_out":
                    ntimed_out+=1
                    ii+=1
                    # print("Gurobi taking more than 3 minutes in some iteration")
                if alg_fail_main:
                    print("failed run")
                    nfail += 1
                else:
                    print("Run complete")
                    time_avg += elapsed_time
                    TIME_AVG_ARR.append(elapsed_time)
                    time_avg_std_dev = np.std(TIME_AVG_ARR)
                    total_iter_avg += np.sum(niterations)
                    total_ILP_avg += np.sum(nf_iterations)
                    ii+=1
                itotal+=1
            
    if ntimed_out < Niter:
        total_iter_avg = total_iter_avg/(Niter-ntimed_out-nfail)
        time_avg = time_avg/(Niter-ntimed_out-nfail)
        
        # fail_avg = nfail/(Niter-ntimed_out)    
    else:
        time_avg = 300.0 # Time out limit on the MILP solver.
        fail_avg = None
    return time_avg, ntimed_out, total_iter_avg, total_ILP_avg, time_avg_std_dev

# Functions for running repeated experiments on gridworlds:
# Running 10 iterations for every experiment configuration:
# Returns average time taken, no. of simulations that timed out,  
# and avg. total_no_iterations across runs that didn't fail or time out, and percentage of failed instances (due to algorithm not being complete)
def run_iterations_SAPs(iM, iN, inP, Niter, allow_diag, KEY):
    time_avg = 0
    ntimed_out = 0
    total_iter_avg = 0 # Total no. of while loops running this algorithm
    nfail = 0
    ii = 0
    nstates = iM*iN
    total_ILP_avg = 0 # Total no. of ILP iterations for this algorithm
    itotal = 0
    TIME_AVG_ARR = []
    while ii < Niter and itotal<100:  # Parsing through iterations
        ns = inP + 1 # Total no. of samples: inP + goal
        G = construct_graph(iM, iN, allow_diag) # grids
        # G = construct_random_graph(n, m)
        props, skip_graph = generate_valid_config_SAPs(G, ns, nstates)
        
        # Enter only valid propositions:
        if not skip_graph:
            t = time.time()
            print("Start run: "+str(ii))
            Q0 = props[0]
            C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main,skip_iter = remove_edges(G, props, Q0,KEY)
            if not skip_iter:
                elapsed_time = time.time() - t # Time that has elapsed
                
                
                if discard=="time_out":
                    ntimed_out+=1
                    ii+=1
                    # print("Gurobi taking more than 3 minutes in some iteration")
                if alg_fail_main:
                    print("failed run")
                    nfail += 1
                else:
                    print("Run complete")
                    time_avg += elapsed_time
                    TIME_AVG_ARR.append(elapsed_time)
                    
                    total_iter_avg += np.sum(niterations)
                    total_ILP_avg += np.sum(nf_iterations)
                    ii+=1
                itotal+=1
        # if iM == 4 and inP == 3:
        #     print("Pause")
    if ntimed_out < Niter:
        total_iter_avg = total_iter_avg/(Niter-ntimed_out-nfail)
        time_avg = time_avg/(Niter-ntimed_out-nfail)
        time_avg_std_dev = np.std(TIME_AVG_ARR)
        assert(time_avg == np.mean(TIME_AVG_ARR))
        # fail_avg = nfail/(Niter-ntimed_out)    
    else:
        time_avg = 300.0 # Time out limit on the MILP solver.
        fail_avg = None
    return time_avg, ntimed_out, total_iter_avg, total_ILP_avg, time_avg_std_dev

# Generates valid configuration for all SAPs:
def generate_valid_config_SAPs(G, ns, nstates):
    skip_iter = True
    # generate a valid configuration
    props, skip_graph = generate_valid_config(G, ns, nstates)
    if not skip_graph:
        count_max = 25 # Try 25 times or else find a new graph to synthesize propositions on
        count = 0
        while count < count_max:
            props_lst = [p[0] for p in props]
            KEY = "SAP"
            # Generate all SAPs and verify that it satisfies it for all SAPs:
            A_combos, SAP_k_list, total_combos = get_SAP_constraints(G, props_lst, KEY, Q0=None)
            min_poss_flow = compute_min_poss_flow(G, props_lst)
            for Af in A_combos:
                for Af_set in Af:
                    if len(Af_set) == min_poss_flow:
                        skip_iter = False
                        break
                if not skip_iter:
                    break
            if not skip_iter:
                break
            else:
                count+=1
    else:
        skip_iter = True
    
    return props, skip_iter
# iterations: Random graphs:
def run_iterations_random_graph(n, m, inP, Niter, allow_diag, KEY):
    time_avg = 0
    ntimed_out = 0
    total_iter_avg = 0 # Total no. of while loops running this algorithm
    nfail = 0
    ii = 0
    nstates = n
    total_ILP_avg = 0 # Total no. of ILP iterations for this algorithm
    itotal = 0
    while ii < Niter and itotal<100:  # Parsing through iterations
        ns = inP + 1 # Total no. of samples: inP + goal 
        # G = construct_graph(iM, iN, allow_diag) # grids
        G = construct_random_graph(n, m)
        props, skip_graph = generate_valid_config(G, ns, nstates)
        
        if not skip_graph:
            # Enter only valid propositions:
            t = time.time()
            print("Start run: "+str(ii))
            Q0 = props[0]
            C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main, skip_iter = remove_edges(G, props, Q0,KEY)
            if not skip_iter:
                elapsed_time = time.time() - t # Time that has elapsed
                print("Run complete")
                time_avg += elapsed_time
                if discard=="time_out":
                    ntimed_out+=1
                    ii+=1
                    # print("Gurobi taking more than 3 minutes in some iteration")
                if alg_fail_main:
                    print("failed run")
                    nfail += 1
                else:
                    total_iter_avg += np.sum(niterations)
                    total_ILP_avg += np.sum(nf_iterations)
                    ii+=1
                itotal+=1
            else:
                print("Skipped")
            if ntimed_out < Niter:
                total_iter_avg = total_iter_avg/(Niter-ntimed_out-nfail)
                time_avg = time_avg/(Niter-ntimed_out-nfail)
                # fail_avg = nfail/(Niter-ntimed_out)    
            else:
                time_avg = 300.0 # Time out limit on the MILP solver.
                fail_avg = None
    return time_avg, ntimed_out, total_iter_avg, total_ILP_avg

# Given a list of 3 propositions: (p1, p2, p3), and that each proposition has 3 neighbors with flows going both ways. 
# Then, the maximum flow has to be 1 since the middle proposition can have atmost 1 unit of flow
def compute_min_poss_flow(G, props):
    m = 1000 # Very large for this.
    for k in range(1, len(props)-1):
        out_nodes = list(G.successors(props[k]))
        in_nodes = list(G.predecessors(props[k]))
        nout = len(out_nodes)
        nin = len(in_nodes)
        common_nodes = [x for x in out_nodes if x in in_nodes]
        ncommon = len(common_nodes)
        # Split between nout and nin:
        nout_only = (nout - ncommon) + ncommon//2
        nin_only = (nin - ncommon) + ncommon//2
        mk = min(nout_only, nin_only)
        if mk < m:
            m = mk
    return m

# =================================== Generating a valid proposition configuration ====================== #
def generate_valid_config(G, ns, nstates):
    node_num = [ii+1 for ii in range(nstates)]    # Numbering nodes
    assert(ns < nstates) 
    prop_gen = False
    MAX_RUNS = 25
    run = 0
    skip_iter = False
    while not prop_gen:
        props0_num = random.sample(node_num, 1)
        if props0_num != []:
            props0 = "n" + str(props0_num[0])
            props = [[props0]]
             # Propositions labeled as 0
            l0 = nx.single_source_shortest_path_length(G, props0)
            nodes_farther_than_ns = []    # keeping tracj of all nodes atleast ns away
            for key, val in l0.items():   # iterating through all items of l0 to search for paths atleast ns away
                if val >= ns:           # Might have to fix this
                    nodes_farther_than_ns.append(key)
            if nodes_farther_than_ns == []:     # Search for a different initial condition
                prop_gen=False
            else:
                propg = random.choice(nodes_farther_than_ns) # Choose a random goal
                path_p0_g = nx.shortest_path(G, props0, propg) # Default option
                shortest_paths_p0_g = nx.all_shortest_paths(G, props0, propg)
                for candidate_path in shortest_paths_p0_g:
                    select = random.choice([True, False]) # Random generator to select a given shortest path:
                    if select:
                        path_p0_g = candidate_path.copy()
                        break
                # Selecting propositions:
                sample_from_path = [path_p0_g[ii] for ii in range(1,len(path_p0_g)-1)] # Exclude end points
                assert(len(sample_from_path) > ns-2)
                sorted_sample = [[sample_from_path[i]] for i in sorted(random.sample(range(len(sample_from_path)), ns-2))] # Sampling points in order
                props += sorted_sample # Adding to initial condition props0
                props += [[propg]]
                prop_gen = True
        run+=1
        if run>25:
            skip_iter = True
            break
    return props, skip_iter
# =================================== Verifying that input is valid ======================================================= #
# This takes too long to run!
# Verify valid configuration: returns true if constraints can be synthesized to find a valid trajectory
# through sequence of propositions or returns false otherwise.
def verify_config(G, props):
    valid = False # Default
    nP = len(props)
    assert(nP>=2)
    found_path = np.zeros((nP-1,1)) # Placeholder; turned to 1 if there is some path from pi to pi+1
    for ii in range(nP-1):
        st_node = props[ii][0]
        end_node = props[ii+1][0]
        st_paths = nx.all_simple_paths(G, st_node, end_node)
        for path in st_paths:
            other_props_present = check_props_path(st_node, end_node, props, path)
            if other_props_present==False:
                found_path[ii]=1
                break
    if found_path.all()==1:
        valid = True
    return valid

# Minor helper function for the verify_config function: Checks if a member of list props (that is not s or t) is in path :
def check_props_path(s,t,props,path):
    other_props_present = False # No other props on path
    props_wo_st = [pi for pi in props if (pi[0]!=s and pi[0]!=t)]
    for pi in props:
        if pi[0] in path:
            other_props_present=True
            break          
    return other_props_present

# ======================================= Helper functions for constructing graphs and converting to directed graphs ============================= #
# Construct a random graph:
# n nodes and M edges
def construct_random_graph(n, m):
    max_edges = n*(n-1)/2
    if m > max_edges:
        m = max_edges
    G = gnm_random_graph(n, m, seed=None, directed=True)
    while not nx.is_weakly_connected(G):
        G = gnm_random_graph(n, m, seed=None, directed=True)
    mapping = {}
    for node in list(G.nodes):
        mapping[node] = "n"+str(node+1)
    G = nx.relabel_nodes(G, mapping)
    for e in list(G.edges):
        u = e[0]
        v = e[1]
        G.remove_edge(u,v)
        G.add_edge(u,v,capacity=1.0)
    return G
# Constructing a gridworld:        
def construct_graph(M,N, allow_diag):
    G = nx.DiGraph()
    nodes = ["n"+str(ii+1) for ii in range(M*N)]
    G.add_nodes_from(nodes)
    G = construct_grid(G, M, N, allow_diag)
    return G
# Function to introduce auxiliary vertices for undirected edges in G to be replaced by directed edges:
# Returns a directed graph and the number of auxiliary vertices. If G is fully undirected, aug_v_count = |G.edges|
def to_directed(G):
    Gdir = G.copy()
    aux_v_count=0
    for e in G.edges:
        u = e[0] # start
        v = e[1] # end
        # Check if reverse edge is also in the graph:
        if (e in Gdir.edges and (v,u) in Gdir.edges): 
            aux_v = "an"+str(aux_v_count)
            Gdir.remove_edge(v,u)
            Gdir.add_edge(v, aux_v, capacity=1.0)
            Gdir.add_edge(aux_v, u, capacity=1.0)
            aux_v_count+=1
    return Gdir, aux_v_count

# ================================================= Post-processing directed graphs with auxiliary vertices ======================================= # 
# Function to post-process cuts and get rid of auxiliary vertices:
def post_process_cuts(C, Gdir):
    postC = C.copy()
    for e in C:
        if e[0][0:2]=="an":
            pred_aux = list(Gdir.predecessors(e[0])) 
            assert(len(pred_aux) == 1) # Sanity check
            pred = pred_aux[0]
            actual_edge = (pred, e[1])
            postC.append(actual_edge)
            postC.remove(e)
        if e[1][0:2]=="an":
            succ_aux = list(Gdir.successors(e[1])) 
            assert(len(succ_aux) == 1) # Sanity check
            succ = succ_aux[0]
            actual_edge = (e[0], succ)
            postC.append(actual_edge)
            postC.remove(e)
    if C!=[]:   # Sometimes there are no cuts
        assert(postC != [])   
    return postC

# Post-processing along with histroy of cuts. This is for plotting purposes:
def process_history(Chist, Gdir):
    nC = len(Chist)
    post_Chist = Chist.copy()
    
    for ii in range(nC):
        Chisti = Chist[ii]
        post_Chist[ii] = post_process_cuts(Chisti, Gdir)
    return post_Chist