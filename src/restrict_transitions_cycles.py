#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:13:18 2020

@author: apurvabadithela
"""


# Dec 7, 2020
# Apurva Badithela
# Script to generate cuts to a graph that handles cycles:

# Import functions:
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
from networkx.algorithms.flow.utils import build_residual_network
from src.milp_functions import find_SAP_k, find_aug_paths, get_SAP_constraints, get_all_SAPs, keep_aug_paths_corrected, cut_aug_paths_corrected, cut_aug_paths_eff, cut_aug_paths, milp_cycles, keep_aug_paths, construct_milp_params, find_nkeep, augment_paths, sdistance, construct_c


# Main functions for remove_edges:

# Function to remove "q0_aug_nodes" and process cuts:
def process_Q0_cuts(Gq0, aug_path_nodes, C):
    Cnew = []
    for e in C:
        if (e[1]=="q0_aug_nodes"):
            for anode in aug_path_nodes:
                candidate_edge = (e[0], anode)
                if candidate_edge in Gq0:
                    Cnew.append(candidate_edge)
        elif (e[0]=="q0_aug_nodes"):
            for anode in aug_path_nodes:
                candidate_edge = (anode, e[1])
                if candidate_edge in Gq0:
                    Cnew.append(candidate_edge)
        else:
            Cnew.append(e)
    return Cnew

# Initial condition is the first proposition:
# No initial conditions to worry about.    
def remove_edges(G, props, Q0, KEY):
    Gin = G.copy()
    alg_fail_main = False
    C = []
    n = len(props)-1 # No. of propositions to cover + goal + initial condition
    assert(n>1)
    i = n-1 # For indexing
    Chist = [] # History of cuts
    discard=False
    niterations = []  # For a given set of paths to cut, niterations keeps track of outer while loop
    nf_iterations = [] # Number of iterations keeping track of flow
    nkeep = 0
    
    # Setting edges to constrian pi:
    C, Chisti, discard_i, nkeep, niter, nf_iter, skip_iter = remove_pi_edges_corrected_cycles(Gin, props, C, KEY) # Input Gf already has input C edges removed from it

    if discard_i == "timeout" or discard_i == "infeasible":
        discard = discard_i
        alg_fail_main=True
    niterations.append(niter) # No. of iterations of the while loop
    nf_iterations.append(nf_iter) # In each iteration of the outer while loop, how many iterations of the inner flow loop are necessary
    Gin = G.copy() # Copying the whole original graph
    Gin.remove_edges_from(C) # Removing all the edges found by remove_pi_edges
    Chist.extend(Chisti) # Appending cuts for the ith node
    return C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main, skip_iter

def remove_edges_corrected_cycle(G, props, Q0=None):
    Gf, node_q0, find_Q0 = check_Q0(G, Q0)
    Gin = Gf.copy()
    alg_fail_main = False
    C = []
    n = len(props)-1 # No. of propositions to cover + goal
    assert(n>0)
    i = n-1 # For indexing
    Chist = [] # History of cuts
    discard=False
    niterations = []  # For a given set of paths to cut, niterations keeps track of outer while loop
    nf_iterations = [] # Number of iterations keeping track of flow
    nkeep = 0
    if i > 0 and n>=2:  # so that it's not just p1 and g
        # Setting edges to constrian pi:
        C, Chisti, discard_i, nkeep, niter, nf_iter, alg_fail = remove_pi_edges_corrected_cycles(Gin, props, C) # Input Gf already has input C edges removed from it
        if alg_fail:
            alg_fail_main = True
        if discard_i:
            discard = discard_i
        niterations.append(niter) # No. of iterations of the while loop
        nf_iterations.append(nf_iter) # In each iteration of the outer while loop, how many iterations of the inner flow loop are necessary
        Gin = Gf.copy() # Copying the whole original graph
        Gin.remove_edges_from(C) # Removing all the edges found by remove_pi_edges
        Chist.extend(Chisti) # Appending cuts for the ith node
        # print(C)
    print("Finding Initial Conditions")
    # When i = 0, its time for constraints to be placed for p1, and if find_Q0 =1, to find the set of initial conditions
    if not (alg_fail_main or discard):
        if find_Q0:
            Q0, aug_path_nodes = pre_p1(Gin, props) # Finding predecessors of p1 (p[0]) not on augmenting paths from p1 to p2
            Gq0, node_q0, find_Q0 = check_Q0(Gin, Q0)
        else:
            Gq0 = Gin.copy()
            aug_path_nodes = all_aug_path_nodes(Gin, props)
            p1 = props[0].copy()
            if p1[0] in aug_path_nodes:
                aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
            
        # Reduce q0 to p1 weight to 1, if it exists: Why?
        if ((node_q0, props[0][0]) in Gq0.edges):
            errant_edge = (node_q0, props[0][0])
            Gq0.remove_edge(*errant_edge)
            Gq0.add_edge(errant_edge[0], errant_edge[1], capacity=1.0)
            
        assert(find_Q0 == 0) # Now, that initial condition points have been found, this should be 0
        # Representing all augmenting paths with one node
        Gq0.add_node("q0_aug_nodes")
        for q in aug_path_nodes:
            Gq0.add_edge(q, "q0_aug_nodes") # Edge considered to have infinite capacity
        aug_nodes = "q0_aug_nodes"
        
        if(node_q0 != props[0][0]): # Don't find cut edges if pi-1 is not the same as pi.
            p0 = [[node_q0], props[0], [aug_nodes]]
            C, Chist0, discard_i, nkeep_q0, niter, nf_iter, alg_fail = remove_pi_edges_Q0_corrected_cycles(Gq0, p0, 1, C, Q0)
            if alg_fail:
                alg_fail_main = True
            elif discard_i:
                discard = discard_i
                alg_fail_main = True
            else:
                C = process_Q0_cuts(Gq0, aug_path_nodes, C) # processing cuts
                Chist_new0 = [] # Processed q0 nodes with clean edges only
                for Cii in Chist0:
                    Cnew_ii = process_Q0_cuts(Gq0, aug_path_nodes, Cii)
                    Chist_new0.append(Cnew_ii)
                Chist.extend(Chist_new0)
                niterations.append(niter) # No. of iterations of the while loop
                nf_iterations.append(nf_iter) # In each iteration of the outer while loop, how many iterations of the inner flow loop are necessary
        else:
            niterations[0] = 0 # node p1 is the same as q0
    else:
        C = []
        Q0 = []
        Chist = []
        nkeep = []
        niterations = []
    return C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main

# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges_corrected_cycles(G, props, C, KEY):
    n = len(props)-1
    Gk = G.copy()
    Chisti = []
    discard = False
    Pcut, n_paths_to_cut = cut_aug_paths_corrected(G, props)    
    
    n_iterations = 0   
    MAX_ITER = 100    
    nf_iter = []
    no_cuts = True
    while np.sum(n_paths_to_cut) > 0:
        no_cuts = False  # Flag to skip an iteration if there's no work to be done
        # Cycle constraints: Get all different possible combinations of flows and SAPs:
        props_lst = [p[0] for p in props]
        print("Finding combinations:")
        A_combos, SAP_k_list, total_combos = get_SAP_constraints(G, props_lst, KEY, Q0=None)
        fopt_max = 0
        
        # Getting different flow paths:
        # SAP_all, nSAP_all, total_SAP_combos, coeff = get_all_SAPs(G, props_lst)
        # print("Found all pairs of SAPs")
        # SAP_k_list = find_SAP_k(G, props_lst, SAP_all, nSAP_all, total_SAP_combos, coeff)
        # Total flow:
        nf_count = 0 # How many iterations of each A_combos do you count?
        print("iteration: ")
        print(n_iterations)
        paths_cut = False # Have all paths in this while iteration been cut with max flow: True/False
        for k in range(total_combos):
            A_f_set = A_combos[k] # A full flow w/o cycles
            Pkeep = SAP_k_list[k]
            MC_keep = find_aug_paths(Gk, Pkeep, props)
            A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
            # min_poss_flow = min(n_keep)
            tracker = 0
            min_poss_flow = compute_min_poss_flow(Gk, props_lst, n_keep)
            for A_f in A_f_set:
                ones = np.ones((len(A_f[0]),))
                ones_f = np.ones((len(A_f),))
                D_f = (A_f @ ones).T # np.array
                newC, fopt, timed_out, feas = milp_cycles(A_cut, n_cut, A_keep, D_keep, n_keep, D_f, A_f, e) # Getting new cuts from the MILP
                # parameter to keep track of updates:
                tracker += 1
                if tracker%1000 == 0:
                    print(tracker)
                # Updating the max. fopt value:
                if ones_f @ fopt > fopt_max:
                    fopt_max = ones_f @ fopt
                    newC_opt = newC.copy()
                
                # Breaking out if the max value has been achieved (i.e equal to the min.possible flow value)
                # then break out of both loops
                if ones_f @ fopt == np.array(min_poss_flow):
                    paths_cut = True
                    break
                nf_count += 1
            if paths_cut:  # Breaking out of outer loop if a satisfying assignment has been found
                break 
        nf_iter.append(nf_count) # Keeping track of how many runs in each iteration    
        n_iterations+=1 # Adding no. of iterations
        if timed_out==True:
            discard = "time_out"
            break
        if feas==True:
            discard="infeasible"
            break
        if n_iterations > MAX_ITER:
            discard = "time_out"
            break
        C = update(G, C, newC) # Updating cuts
        Chisti.append(newC)
        Gi = G.copy()
        Gk = G.copy()
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths_corrected(Gi, props)    
        # Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)
    if no_cuts:
        nkeep = 0
        skip_iter = True
    else:
        nkeep = find_nkeep(Pkeep)
        skip_iter = False
    return C, Chisti, discard, nkeep, n_iterations, nf_iter, skip_iter

# Processes Q0_edges being cut:
def  update_Q0_edges(Pcut, newC):
    newC_adj = []
    # Checking if edges in newC have "q0_aug_nodes" as the end vertex in an edge
    for e in newC:
        if e[1]=="q0_aug_nodes":
            for paths in Pcut:
                for p in paths:
                    if e[0] in p:   # If the first node is in the path p
                        e0_idx = p.index(e[0])
                        e0_pred = p[e0_idx-1]
                        candidate_edge = (e0_pred, e[0])
                        if candidate_edge not in newC_adj:
                            newC_adj.append(candidate_edge)
        else:
            newC_adj.append(e)
    return newC_adj

# Given a list of 3 propositions: (p1, p2, p3), and that each proposition has 3 neighbors with flows going both ways. 
# Then, the maximum flow has to be 1 since the middle proposition can have atmost 1 unit of flow
def compute_min_poss_flow(G, props, n_keep):
    m = min(n_keep)
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

# Checks of initial condition set is provided; if not signals to the main 
# restrict_transitions function that it needs to be created
def check_Q0(G, Q0):
    Gf = G.copy()
    if Q0 is None:
        find_Q0 = 1
        node_q0 = None
    else:
        find_Q0 = 0
        assert(q0 in G.nodes for q0 in Q0)
        if len(Q0)==1:
            node_q0 = Q0[0]
        else:
            Gf.add_node("q0")
            for q0 in Q0:
                Gf.add_edge("q0", q0) # Edge considered to have infinite capacity
            node_q0 = "q0"
    return Gf, node_q0, find_Q0


# Function to remove edges for proposition pi, and adding newly removed edges to C
def remove_pi_edges_Q0_corrected_cycles(G, props, i, C, Q0):
    n = len(props)-1
    Gi = G.copy()
    Gk = G.copy()
    Gi.remove_node(props[i][0]) 
    discard = False
    Pcut, n_paths_to_cut = cut_aug_paths_corrected(G, props)    
    # Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)   
    Chist0 = [] 
    niterations = 0 
    nf_iter = []     
    MAX_ITER = 100            
    while np.sum(n_paths_to_cut) > 0:
        # Cycle constraints: Get all different possible combinations of flows and SAPs:
        props_lst = [p[0] for p in props]
        A_combos = get_SAP_constraints(G, props_lst, Q0) # Calling the SAP_constraints function for Q0
        fopt_max = 0
        nf_count = 0
        
        SAP_all, nSAP_all, total_SAP_combos, coeff = get_all_SAPs(G, props_lst)
        SAP_k_list = find_SAP_k(G, props_lst, SAP_all, nSAP_all, total_SAP_combos, coeff)
        # Total flow:
        for k in range(total_SAP_combos):
            A_f_set = A_combos[k]
            Pkeep = SAP_k_list[k]
            MC_keep = find_aug_paths(G, Pkeep, props)
            A_cut, n_cut, A_keep, D_keep, n_keep, cost, e = construct_milp_params(Pcut,Pkeep, MC_keep, n)
            for A_f in A_f_set:
                ones = np.ones((len(A_f[0]),))
                ones_f = np.ones((len(A_f),))
                D_f = (A_f @ ones).T # np.array
                newC, fopt, timed_out, feas = milp_cycles(A_cut, n_cut, A_keep, D_keep, n_keep, D_f, A_f, e) # Getting new cuts from the MILP
                if ones_f @ fopt > fopt_max:
                    fopt_max = ones_f @ fopt
                    newC_opt = newC.copy()
                nf_count += 1
        nf_iter.append(nf_count) # Keeping track of how many runs in each iteration   
        niterations+=1 # Adding iterations
        if timed_out==True:
            discard = "time_out"
            break
        if feas==True:
            discard="infeasible"
            break
        if niterations > MAX_ITER:
            discard = "time_out"
            break
        newC_adj = update_Q0_edges(Pcut, newC) # Processing edges being cut if they include q0_aug_nodes
        C = update(G, C, newC_adj) # Updating cuts
        Chist0.append(newC_adj)
        
        # Prepare for next iteration of cuts for pi:
        Gi = G.copy()
        Gk = G.copy()
        # Gi.remove_node(props[i][0]) 
        Gi.remove_edges_from(C)
        Gk.remove_edges_from(C)
        Pcut, n_paths_to_cut = cut_aug_paths_corrected(Gi, props)    
        Pkeep, MC_keep, alg_fail = keep_aug_paths_corrected(Gk, props)
    nkeep = find_nkeep(Pkeep)
    return C, Chist0, discard, nkeep, niterations, nf_iter, alg_fail

# --------------------------- HELPER functions ----------------------------- #
# Remove q0-->v edges where v is in Q0: 
# Removing vectors associated with q0 ---> v
def remove_q0_edges(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e):
    # Find q0-->v indices in e:
    q0_idx_e = []
    for ii in range(len(e)):
        ei = e[ii]
        if(ei[0]=="q0"):
            q0_idx_e.append(ii)
    if(q0_idx_e):
        new_e = [e[ii] for ii in range(len(e)) if ii not in q0_idx_e]
        nA_cut = []
        nA_keep = []
        nD_keep = []
        nn_cut = []
        nn_keep = []
        for jj in range(len(n_cut)):
            njj = n_cut[jj]
            nnjj = njj
            new_Ai = np.array([])
            Ai = A_cut[jj].copy()
            # Converting to 2d array
            if Ai.ndim == 1:
                Ai=Ai[:,np.newaxis]
            for k in range(len(e)):
                if k not in q0_idx_e:
                    if new_Ai.size != 0:
                        new_Ai = np.column_stack((new_Ai, Ai[:,k]))
                    else:
                        new_Ai = Ai[:,k]
            # Verify there is still a path to cut:
            for r in range(njj):
                if not new_Ai[r,:].any():
                  nnjj-=1
                
            # Remove rows with zeros:
            if new_Ai.ndim>1:
                new_Ai = new_Ai[~np.all(new_Ai == 0, axis=1)]
            if nnjj != 0:
                nn_cut.append(nnjj)
                nA_cut.append(new_Ai)
        
        for jj in range(len(n_keep)):
            njj = n_keep[jj]
            nnjj = njj
            new_Ai = np.array([])
            new_Di = np.array([])
            Ai = A_keep[jj].copy()
            Di = D_keep[jj].copy()
            # Converting to 2d array
            if Ai.ndim == 1:
                Ai=Ai[:,np.newaxis]
            # Converting to 2d array
            if Di.ndim == 1:
                Di=Di[:,np.newaxis]
            for k in range(len(e)):
                if k not in q0_idx_e:
                    if new_Ai.size != 0:
                        new_Ai = np.column_stack((new_Ai, Ai[:,k]))
                    else:
                        new_Ai = Ai[:,k]
            new_Di = new_Ai @ np.ones((len(new_e),1))
                    
            # Verify there is still a path to cut:
            #for r in range(njj):
            #    if not new_Ai[r,:].any():
            #      nnjj-=1
            # Remove rows with all zeros:
            #if new_Ai.ndim>1:
            #    new_Ai = new_Ai[~np.all(new_Ai == 0, axis=1)]
            #    new_Di = new_Di[~np.all(new_Di == 0, axis=1)]
            
            if nnjj!=0:      
                nn_keep.append(nnjj)
                nA_keep.append(new_Ai)
                nD_keep.append(new_Di)
        new_cost = construct_c(nn_keep)
    else:
        nA_cut = A_cut.copy()
        new_e = e.copy()
        nn_cut = n_cut.copy()
        nA_keep = A_keep.copy()
        nD_keep = D_keep.copy()
        nn_keep = n_keep.copy()
        new_cost = construct_c(nn_keep)
    return nA_cut, nn_cut, nA_keep, nD_keep, nn_keep, new_cost, new_e

# Updating cuts:
def update(G, C, newC):
    for new_cut_edge in newC:
        assert(new_cut_edge not in C) # sanity check to make sure we're not cutting edges that have already been cut
    if newC:
        R = build_residual_network(G, capacity="capacity")
        for edge in newC:
            if R[edge[0]][edge[1]]['capacity'] == 1:
                C.append(edge)
    return C

# Returns nodes on all augmented paths:
def all_aug_path_nodes(G, props):
    aug_path_nodes = []
    for ii in range(len(props)-1):
        pi = props[ii]
        pi_next = props[ii+1]
        P = augment_paths(G, pi, pi_next)
        if aug_path_nodes == []:
            new_aug_path_nodes = list(set().union(*P)) # Nodes on paths P
            aug_path_nodes = new_aug_path_nodes.copy()
        else:
            new_aug_path_nodes = list(set().union(*P))
            aug_path_nodes = add_nodes(aug_path_nodes, new_aug_path_nodes)
    assert(aug_path_nodes!=[])
    return aug_path_nodes

# Adding nodes from nodes_closer_goal to aug_path_nodes without repeating elements:
def add_nodes(aug_path_nodes, nodes_closer_goal):
    new_aug_path_nodes = aug_path_nodes.copy()
    for node in nodes_closer_goal:
        if node not in aug_path_nodes:
            new_aug_path_nodes.append(node)
    return new_aug_path_nodes


# Function returning predecessors of p1 that are not on aug paths from p1 to p2, p2 to p3, ...
def pre_p1(G, props):
    aug_path_nodes = all_aug_path_nodes(G, props)
    # p2 = props[1].copy() # Second proposition on the graph
    # g = props[-1].copy() # Goal node on the graph
    # nodes_closer_goal = sdistance(G, p2, g)
    # aug_path_nodes= add_nodes(aug_path_nodes, nodes_closer_goal)
    G0 = G.copy()
    # Remove nodes that are closer to the goal than p2:
    p1 = props[0].copy()
    if p1[0] in aug_path_nodes:
        aug_path_nodes.remove(p1[0]) # Don't want to remove the source vertex
    else:
        print("Something fishy:")
        print(aug_path_nodes)
        print(p1[0])
    G0.remove_nodes_from(aug_path_nodes)
    ancestor_tree = nx.dfs_tree(G0.reverse(), source=p1[0]).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    pred_p1_init = list(ancestor_tree.nodes)
    pred_p1=pred_p1_init.copy()
    # Removing auxiliary vertices:
    for v in pred_p1_init:
        if (v[0:2]=="an"):
            pred_p1.remove(v)
    assert(pred_p1 != [])
    return pred_p1, aug_path_nodes
