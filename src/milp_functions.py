#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:28:18 2020

@author: apurvabadithela
"""


# File contains MILP functions that are used by the restrict_transitions_simplified.py class

import numpy as np
import time
import itertools
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
from src.preprocess_constraints_SAP import SAP_sets
from src.preprocess_constraints_ALL import ALL_sets
from numpy import prod

# ================================== Functions to handle cycle constraints ========================================== #
# Function for MiLP to accommodate cycles:
# Inputs: Matrices: A_cut, A_f, A_keep; Arrays: D_keep, D_f; misc: n_cut, n_keep, e
# Outputs: edges to remove: newC, optimal flow vector: fopt, timed_out or infeasibility optimization variables
def milp_cycles(A_cut, n_cut, A_keep, D_keep, n_keep, D_f, A_f, e):
    nc = len(A_f[0])
    ne = len(e)
    nf = len(A_f)
    epsilon = 0.5 # Factor that keeps b variables to 1
    newC = []
    fopt = 0
    timed_out = False
    feas = False
    try:
        # Create a new model
        m = gp.Model("milp")
        m.setParam('OutputFlag', 0)  # Also dual_subproblem.params.outputflag = 0
        m.setParam(GRB.Param.TimeLimit, 300.0) # Setting time limit: 5 min
        # Create variables: 
        x = m.addMVar(ne, vtype=GRB.BINARY, name="x")
        b = m.addMVar(nc, vtype=GRB.BINARY, name="b")
        f = m.addMVar(nf, vtype=GRB.BINARY, name="f")
        ones = np.ones((nf,))
        m.params.threads = 4
        # Set objective: c.T*b; minimizing cuts to augmenting paths pj to pj+1
        m.setObjective(ones @ f, GRB.MAXIMIZE)
        
        # Minimizing the number of cuts as well:
        xones = np.ones((ne,), dtype=int)
        
        jcut_idx = 0
        for jj in range(len(n_cut)):
            njj = n_cut[jj]
            Aj = A_cut[jj]
            rhs = np.ones((njj,), dtype=int)
            assert(len(Aj)==njj) # Sanity check
            assert(len(Aj[0])==ne) # Sanity check
            m.addConstr(Aj @ x >= rhs, name="c1_"+str(njj))
            jcut_idx += njj    
        
        # Add constraints: Ajj+1 x <= Djj+1 b; bjj+1 <= Ajj+1x
        jkeep_idx = 0
        for jj in range(len(n_keep)):
            njj = n_keep[jj]
            Aj = A_keep[jj]
            Dj = D_keep[jj]
            rhs = np.zeros((njj,), dtype=int)
            m.addConstr(Aj @ x <= np.diag(Dj[:,0]) @ b[jkeep_idx: jkeep_idx+njj], name="c2i_"+str(njj))
            assert(len(Aj @ np.ones((ne,), dtype=int)) == njj) # Sanity check
            assert(len(np.diag(Dj[:,0]) @ np.ones((njj,), dtype=int)) == njj) # Sanity check
            rhs = (1-epsilon)*np.ones((njj,), dtype=int)
            # Getting rid of the epsilon constraint:
            # m.addConstr(b[jkeep_idx: jkeep_idx+njj] - Aj @ x <= rhs, name="c2ii_"+str(njj))
            m.addConstr(b[jkeep_idx: jkeep_idx+njj]  <= Aj @ x, name="c2ii_"+str(njj))
            ones = np.ones((njj,), dtype=int)
            # lhs = np.dot(ones, b[j_idx: j_idx+njj])
            # m.addConstr(ones @ b[jkeep_idx: jkeep_idx+njj] <= njj-1)
            jkeep_idx += njj
        
        # Add constraints: D_f@ f <= A_f @ (1-b); A_f@(1-b) - 1@D_f + 1 <= f
        ones_b = np.ones((nc,))
        ones_f = np.ones((nf,))
        m.addConstr(np.diag(D_f) @ f + A_f @ b <= A_f @ ones_b, name="c3")
        m.addConstr(A_f @ ones_b  + ones_f <= f + A_f @ b + D_f @ ones_f, name="c4")
            
        # Optimize model
        m.optimize()
        if(m.status==GRB.INFEASIBLE):
            feas = True
        if(m.status == GRB.TIME_LIMIT):
            timed_out = True

        if feas!=True and timed_out!=True:
            xopt=np.zeros((ne,1))
            bopt = np.zeros((nc,1))
            fopt = np.zeros((nf,1))
            xidx = 0
            bidx = 0
            fidx = 0
            for v in m.getVars():
                if(xidx < ne and bidx < nc):
                    xopt[xidx] = v.x
                    xidx+=1
                elif(xidx==ne and bidx < nc):
                    bopt[bidx] = v.x
                    bidx+=1
                else:
                    fopt[fidx] = v.x
                    fidx += 1
            
            newC = interpret_milp_output(xopt,e)
            # print(newC) 
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    return newC, fopt, timed_out, feas

# The following functions are meant to handle cycle constraints and enforce recursive feasibility:
# For a given graph with source s and sink t, find all constraints representing shortest augmenting paths in G from s to t

# Getting the full flow combination matrices for a set of augmented paths:
def flow_combo_matrix(SAP_k, Q0=None):
    A_prod = []
    num_flows_SAP_k = [len(SAP_k[ii]) for ii in range(len(SAP_k))] # no. of flows for each pair of vertices in SAP_k
    total_combinations = prod(num_flows_SAP_k)
    max_poss_flow = min(num_flows_SAP_k) # Max. possible flow is the minimum one through the pipeline. Each flow combination has a max possible flow given by max_poss_flow
    flow_combos = []
    flow_combos_indices = [] # Indices representing the different flows that were combined to make the master flow.
    for ii in range(1, total_combinations+1):  # For all combinations of pk to pk+1 flows, find a candidate for p1 to pn+1 flow
        candidate_flow, cand_flow_indices, cand_flow_valid = find_candidate_flow(SAP_k, num_flows_SAP_k, ii, Q0)   # A candidate flow is a sequence of flows from pk to pk+1 stitched together
        if cand_flow_valid:
            flow_combos.append(candidate_flow)
            flow_combos_indices.append(cand_flow_indices)
    possible_flow_combos = find_possible_flows(flow_combos_indices, max_poss_flow, flow_combos)
    total_flow_combos = len(possible_flow_combos)
    
    # total_flow_combos: Different possible flows. The only way to handle all sieve out that flow is not through flow paths that have cycles
    max_A_prod = 0 # Finding the maximum flows in the product matrices
    min_A_prod = 0 # Finding the minimum flows in the product matrices
    for ii in range(total_flow_combos):
        P = possible_flow_combos[ii].copy()
        A_prod_ii = construct_flow_matrix(P, num_flows_SAP_k)
        A_prod.append(A_prod_ii)
        if ii == 0:
            min_A_prod = len(A_prod_ii)
        if len(A_prod_ii) < min_A_prod:
            min_A_prod = len(A_prod_ii)
        if len(A_prod_ii) > max_A_prod:
            max_A_prod = len(A_prod_ii)
            # if max_A_prod ==2:
            #     print("Hi")
    return A_prod, max_A_prod, min_A_prod

# Finding if a combination of pk to pk+1 flows forms a valid candidate with no cycles:
# Q0 is the matrix containing initial conditions (optional)
def find_candidate_flow(SAP_k, Nf, ii, Q0=None):
    candidate_flow = [] # Default
    cand_flow_indices = [] # Candidate flow indexed by which flow is active at every step
    is_valid = False # Default
    coeff = build_coeff(Nf) # Coefficient for the product equation given the number of paths we have between propositions
    fids = find_SAP_indices(coeff, ii)  # Which flows are combined to make the flow 
    P = []
    P_indices = []
    for k in range(len(Nf)):
        P.append(SAP_k[k][int(fids[k]-1)]) # SAP_k[k] returns the set of flow paths from pk to pk+1; fids[k] returns the label of the path that is selected as a flow from pk to pk+1
        P_indices.append(fids[k])
    is_valid = check_cycle_in_flow(P, Q0) # Checks if this combination of paths doesn't contain a cycle
    if is_valid:
        candidate_flow = P
        cand_flow_indices = P_indices
    return candidate_flow, cand_flow_indices, is_valid

# path for checking all possible flows:
# Different ways in which a flow = f can be constructed from a set of N different individual candidate flows
# Possible_flow_combos returns the set of all paths 
# Find all the total posssible flows:
# Flow_combos is the variable containing all the flow_combos_paths
def find_possible_flows(combined_flows, max_poss_flow, flow_combos):
    possible_flow_combos = []
    m = min(max_poss_flow, len(combined_flows)) # Whichever is smaller
    combined_flow_idx = [ii for ii in range(len(combined_flows))] # Indices for taking combinations of different flows
    for ii in range(m, 0, -1): # Decreasing backwards.
        for total_flow_idx in itertools.combinations(combined_flow_idx, ii):
            total_flow = tuple([combined_flows[k] for k in total_flow_idx])  # Indices of the paths from 1 to Np.
            flow_paths = [flow_combos[k] for k in total_flow_idx] # Actual paths
            lst_total_flow = list(total_flow)    # list version of total flow
            N = len(lst_total_flow[0])
            paths_chosen = [[] for ii in range(N)] # In a flow configuration, keeps track of all flows that have already been seen. 
            invalid = False # Is the chosen combination of flows invalid? invalid if the same pk to pk+1 flow appears in two different places; Or if two flows form cycles with each other
            stored = False # If the flow is already stored, don't add it to the list
            # Keeping track of all flows that have been added to the list
            for fi in total_flow:
                for k in range(N):
                    fik = fi[k]
                    if fik not in paths_chosen[k]:
                        paths_chosen[k].append(fik)
                    else:
                        invalid=True
                        break
                if invalid:   # break out of the fi in total_flow loop
                    break
            for existing_flow in possible_flow_combos:
                stored = all([elem in existing_flow for elem in lst_total_flow])
                if stored:
                    break
            if (not invalid and not stored):   # if the combination of flows is not invalid, then add it to possible_flow_combos list
                valid_set_flows = check_valid_set_flows(flow_paths)
                if valid_set_flows:
                    possible_flow_combos.append(lst_total_flow)
    return possible_flow_combos

# Checking if a combination of sequence flows is a valid flow and doesnt contain cycles:
# Checked: seems correct
def check_cycle_in_flow(P, Q0=None):
    is_P_good = True # default
    for ii in range(len(P)):
        vP_ii = P[ii][1:-1]
        vP_rest  = []
        if ii < len(P):
            for jj in range(ii+1, len(P)):
                vP_rest.extend(P[jj])
        if "q0_aug_nodes" in vP_rest:
            vP_rest.extend(Q0)           # If there are initial states Q0, adding all nodes connected to vP_rest so that cycles are not missed
        for v in vP_ii:
            if v in vP_rest:
                is_P_good = False
    return is_P_good

# Check if the following set of flow paths are valid together:
# It could be possible that one of the paths has a cycle with another path
# Suppose flow_paths comprises of [f1, f2],where f1 = [x1,x2] and f2 = [x3,x4]. Need to ensure that [x1,x4] and [x2,x3] are also valid flows
# flow_paths: Selected set of flows. 
def check_valid_set_flows(flow_paths):
    valid = True
    N = len(flow_paths[0])
    Nf = [len(flow_paths) for ii in range(N)]
    coeff = build_coeff(Nf)
    total_combinations = prod(Nf)
    for k in range(1, total_combinations+1):
        xids = find_SAP_indices(coeff, k)
        P = []
        for j in range(N):
            flow_k = [f[j] for f in flow_paths]
            sequence_path = flow_k[int(xids[j])-1] # Indexed by 1
            P.append(sequence_path)  # sequence_path is the path between pk and pk+1 that needs to be updated
        is_P_good = check_cycle_in_flow(P)
        if not is_P_good:
            valid = False
            break
    return valid

# Given a particular set of augmenting flows, construct flow combination matrix
# An element of P represents a combination of augmenting paths from p1 to pN.
# Nf is the number of flow paths between each sequence of propositions. The row dimension of A need not be the max possible flow, but rather the length of the P matrix
# A is the matrix with zeros and ones.
def construct_flow_matrix(P, Nf):
    A = np.zeros((len(P), np.sum(Nf)))
    for ii in range(len(P)):
        Pi = P[ii]
        for jj in range(len(Pi)):
            k = int(np.sum(np.array(Nf[0:jj])) + (Pi[jj]-1)) # np.sum(np.array(Nf[0:jj])) keeps track of all the indices already past; and Pi[jj]-1 starts from index 0 for the path labeled by Pi[jj] from pjj to pjj+1 
            A[ii][k] = 1
    return A

# Getting all constraints due to different augmenting paths
# the main function to call for modified flow constraints:
# Q0 is the set of initial conditions to include in the A_prod set
def get_SAP_constraints(G, props, KEY, Q0=None):
    SAP_all, nSAP_all, total_SAP_combos, coeff = get_all_SAPs(G, props, KEY)
    A_combos = []
    pruned_SAP_k = []
    pruned_total = 0
    size_arr = [] # We will use this array to sort the A_combos list in the end. The size of an A_prod matrix will be the length of the array inside it 
    for k in range(1, total_SAP_combos+1):
        xk = find_SAP_indices(coeff, k)
        SAP_k = [] # Set of shortest augemented paths for kth iteration
        for ii in range(len(xk)):
            SAP_k.append(SAP_all[ii][int(xk[ii]-1)]) # Maybe -1 should be inside the xk term
        A_prod, max_Aprod, min_Aprod = flow_combo_matrix(SAP_k, Q0) # Set of product matrices for all possible combinations of flows for the A matrix
        if A_prod:
            A_combos.append(A_prod)
            pruned_SAP_k.append(SAP_k)     
            pruned_total+=1
            size_arr.append(max_Aprod)
    # Sort the combinations matrices so that those with highest flow are listed first:
    sorted_arr = np.argsort(-1*np.array(size_arr))
    A_combos_srt = [A_combos[ii] for ii in sorted_arr]
    pruned_SAP_k_srt = [pruned_SAP_k[ii] for ii in sorted_arr]
    
    # # check:
    # A_combos_srt = A_combos.copy()
    # pruned_SAP_k_srt = pruned_SAP_k.copy()
    return A_combos_srt, pruned_SAP_k_srt, pruned_total

# get_all_SAPs is a good function for finding P_keep
def get_all_SAPs(G,props, KEY):
    SAP_all = []
    nSAP_all = [] # No. of SAPs between each pair of nodes
    total_SAP_combos = 1
    
    # Gathering all shortest possible paths between every two pair of propositions:
    for ii in range(1,len(props)):    # This loop will always go through
        s = props[ii-1]
        t = props[ii]
        Gst = G.copy()
        other_props = [pi for pi in props if (pi!=s and pi!=t)] # All propositions except s and t
        Gst.remove_nodes_from(other_props)  # Same graph as G but without s and t
        if KEY == "SAP":
            SAP = SAP_sets(Gst,s,t)    # Augmenting paths without the other propositions
        elif KEY =="ALL":
            SAP = ALL_sets(Gst,s,t)    # Augmenting paths without the other propositions
        else:
            SAP = SAP_sets(Gst,s,t)    # Augmenting paths without the other propositions
        SAP_all.append(SAP)
        nSAP_all.append(len(SAP))
        total_SAP_combos *= len(SAP)
        
    coeff = build_coeff(nSAP_all)
    return SAP_all, nSAP_all, total_SAP_combos, coeff

# Finding coefficients (1, a1, a2, ..) for the product function given the number of propositions between each pair of propositions: ex: Nf = [2,3,7]
# Means there are 2 paths from p1 to p2, 3 options from p2 to p3, and 7 options from p3 to p4.
def build_coeff(Nf):
    coeff = np.zeros(len(Nf),)
    coeff[0] = 1
    for ii in range(1, len(Nf)):
        coeff[ii] = prod(list(Nf[0:ii]))
    return coeff

# Helper functions:
# Given a function f(x) = x0 + a1*(x1-1) + ... + an*(xn-1), where the coefficients a1, ..., an are defined. 
# Takes as input the identifying number k and the list of coefficients and outputs a list of identifying coefficient indices:
# The output returns xids = [x0, x1, ..., xn]
def find_SAP_indices(coeff, k):    
    xids = np.zeros(len(coeff),)
    sum_k = k
    for ii in range(len(coeff)-1,-1,-1):
        r = sum_k % coeff[ii] # remainder
        q = sum_k // coeff[ii] # quotient
        if r == 0:
            if ii == 0:
                xids[ii] = q
            else: 
                xids[ii] = q 
            sum_k = sum_k - coeff[ii]*(q-1)
        else:
            xids[ii] = q+1
            sum_k = sum_k - coeff[ii]*q
    return xids

# Function to find the SAP_k augmented paths:
def find_SAP_k(G, props_lst, SAP_all, nSAP_all, total_SAP_combos, coeff):
    SAP_k_list = []
    # SAP_all, nSAP_all, total_SAP_combos, coeff = get_all_SAPs(G, props_lst) # need not repeat executing this line
    for k in range(1, total_SAP_combos+1):
        xk = find_SAP_indices(coeff, k)
        SAP_k = [] # Set of shortest augemented paths for kth iteration
        for ii in range(len(xk)):
            SAP_k.append(SAP_all[ii][int(xk[ii]-1)]) # Maybe -1 should be inside the xk term
        SAP_k_list.append(SAP_k)
    return SAP_k_list

# Finds minimum-cut egdes on a graph G with propositions props, and shortest augmenting paths SAP_k
def find_aug_paths(G, Pkeep, props):
    MC_keep = []
    nPkeep = len(Pkeep) # Length of Pkeep
    for jj in range(nPkeep-1, -1, -1):
        MCj = prepare_Gj(G, props, jj, Pkeep[jj], Pkeep) # This assumes there is always a Pkeep for every j to j+1
        MC_keep.append(MCj)
    MC_keep.reverse() 
    return MC_keep
    
# ================================= Original functions ================================================================== # 
# Finding number of augmented paths in final graph:
def find_nkeep(Pkeep):
    ni = [0 for Pi in Pkeep]
    for ii in range(len(Pkeep)):
        Pi = Pkeep[ii]
        ni[ii] = len(Pi)
    nkeep=ni.copy()
    return nkeep

# Function that searches for augmented paths on a graph from source p[j<i] to pi+1 that bypasses waypoint p[i]
 # If Pcut_j is empty, there are no augmenting paths from pj to pi+1
def cut_aug_paths(G, props, i):
    Pcut = []
    n_paths_to_cut = []
    for j in range(i):
        n_paths_to_cut_j = 0
        Pcut_j = augment_paths(G, props[j], props[i+1]) # From pj to pi+1; which is index by pi
        if Pcut_j:
            n_paths_to_cut_j+=len(Pcut_j) # If there are paths to cut, update n_paths_to_cut
        Pcut.append(Pcut_j)
        n_paths_to_cut.append(n_paths_to_cut_j)
    return Pcut, n_paths_to_cut

# Corrected version: Cuts all paths from pj to pi+1
def cut_aug_paths_corrected(Gi, props):
    Pcut = []
    n_paths_to_cut = []
    N = len(props)
    for i in range(1, N-1):
        G= Gi.copy()
        G.remove_node(props[i][0])
        Pcut_new = []
        n_paths_to_cut_ji = []
        for j in range(i):
            # n_paths_to_cut_ji = []
            Pcut_j = augment_paths(G, props[j], props[i+1]) # From pj to pi+1; which is index by pi
            Pcut_j_new = Pcut_j.copy()
            if (j < i-1): # checking if any of pj+1 to pi-1 are on aug path; if so, remove them
                higher_props = [props[k][0] for k in range(j+1, i)]   
                for Pj in Pcut_j:
                    higher_prop_present = any(ph in Pj for ph in higher_props)
                    if higher_prop_present:
                        Pcut_j_new.remove(Pj)
            if Pcut_j_new:
                n_paths_to_cut_ji.append(len(Pcut_j_new)) # If there are paths to cut, update n_paths_to_cut
            
            Pcut_new.append(Pcut_j_new)
        Pcut.extend(list(reversed(Pcut_new)))
        n_paths_to_cut.extend(list(reversed(n_paths_to_cut_ji)))
    return Pcut, n_paths_to_cut

# function that searches for augmented paths and min-cut edges that need to be kept
def keep_aug_paths(G, props, i):
    Pkeep = []
    MC_keep = []
    alg_fail = False
    for jj in range(i+1):
        Pkeep_j = augment_paths(G, props[jj], props[jj+1])            
        try:
            assert(Pkeep_j != []) # Shouldn't be empty. there should always be a path from pj to pj+1; otherwise example is invalid
        except AssertionError as e:
            alg_fail = True
            break
        Pkeep.append(Pkeep_j)
        MCj = min_cut_edges(G, props[jj], props[jj+1])
        MC_keep.append(MCj)
    return Pkeep, MC_keep, alg_fail

# Function that searches for augmented paths for all indices; all at once
def keep_aug_paths_corrected(G, props):
    Pkeep = []
    MC_keep = []
    alg_fail = False
    n = len(props)-1
    for jj in range(n):
        other_props = [p[0] for p in props if (p!=props[jj] and p!=props[jj+1])] # Other props that might be present; want to get rid of
        Gjj = G.copy()
        Gjj.remove_nodes_from(other_props)
        Pkeep_j = augment_paths(Gjj, props[jj], props[jj+1]) 
        Pkeep_j_new = Pkeep_j.copy()
        # Consider only augmented paths not going through other vertices:
        # for Pj in Pkeep_j:
        #     other_prop_present = any(props[k][0] in Pj for k in range(n) if (k!=jj and k!=(jj+1)))
        #     if other_prop_present:
        #         Pkeep_j_new.remove(Pj)           
        try:
            assert(Pkeep_j != []) # Shouldn't be empty. there should always be a path from pj to pj+1; otherwise example is invalid
        except AssertionError as e:
            alg_fail = True
            break
        Pkeep.append(Pkeep_j_new)
        
    nPkeep = len(Pkeep) # Length of Pkeep
    for jj in range(nPkeep-1, -1, -1):
        MCj = prepare_Gj(G, props, jj, Pkeep[jj], Pkeep) # This assumes there is always a Pkeep for every j to j+1
        MC_keep.append(MCj)
    MC_keep.reverse() # Returns MC_keep in the correct order
    return Pkeep, MC_keep, alg_fail

# Main MILP program:
def milp(A_cut, n_cut, A_keep, D_keep, n_keep, cost, e):
    nc = len(cost)
    ne = len(e)
    epsilon = 0.5 # Factor that keeps b variables to 1
    newC = []
    timed_out = False
    feas = False
    try:
        # Create a new model
        m = gp.Model("milp")
        m.setParam(GRB.Param.TimeLimit, 300.0) # Setting time limit: 5 min
        # Create variables: 
        x = m.addMVar(ne, vtype=GRB.BINARY, name="x")
        b = m.addMVar(nc, vtype=GRB.BINARY, name="b")
        m.setParam('OutputFlag', 0)  # Also dual_subproblem.params.outputflag = 0
        m.params.threads = 4
        # Set objective: c.T*b; minimizing cuts to augmenting paths pj to pj+1
        m.setObjective(cost[:,0] @ b, GRB.MINIMIZE)
        
        # Minimizing the number of cuts as well:
        xones = np.ones((ne,), dtype=int)
        # m.setObjective(cost[:,0] @ b + xones @ x, GRB.MINIMIZE) ---> Avoids cycles in most cases
        m.setObjective(cost[:,0] @ b, GRB.MINIMIZE) # ---> Objective is only about cuts and flow
        # Add constraint: Aji+1 x = 1
        jcut_idx = 0
        for jj in range(len(n_cut)):
            njj = n_cut[jj]
            Aj = A_cut[jj]
            rhs = np.ones((njj,), dtype=int)
            assert(len(Aj)==njj) # Sanity check
            assert(len(Aj[0])==ne) # Sanity check
            m.addConstr(Aj @ x >= rhs, name="c1_"+str(njj))
            jcut_idx += njj    
        
        # Add constraint: Ajj+1 x <= Djj+1 b
        jkeep_idx = 0
        for jj in range(len(n_keep)):
            njj = n_keep[jj]
            Aj = A_keep[jj]
            Dj = D_keep[jj]
            rhs = np.zeros((njj,), dtype=int)
            m.addConstr(Aj @ x <= np.diag(Dj[:,0]) @ b[jkeep_idx: jkeep_idx+njj], name="c2i_"+str(njj))
            assert(len(Aj @ np.ones((ne,), dtype=int)) == njj) # Sanity check
            assert(len(np.diag(Dj[:,0]) @ np.ones((njj,), dtype=int)) == njj) # Sanity check
            rhs = (1-epsilon)*np.ones((njj,), dtype=int)
            # Getting rid of the epsilon constraint:
            # m.addConstr(b[jkeep_idx: jkeep_idx+njj] - Aj @ x <= rhs, name="c2ii_"+str(njj))
            m.addConstr(b[jkeep_idx: jkeep_idx+njj]  <= Aj @ x, name="c2ii_"+str(njj))
            ones = np.ones((njj,), dtype=int)
            # lhs = np.dot(ones, b[j_idx: j_idx+njj])
            m.addConstr(ones @ b[jkeep_idx: jkeep_idx+njj] <= njj-1)
            jkeep_idx += njj
    
        # Optimize model
        m.optimize()
        
        if(m.status==GRB.INFEASIBLE):
            feas = True
        if(m.status == GRB.TIME_LIMIT):
            timed_out = True

        if feas!=True and timed_out!=True:
            xopt=np.zeros((ne,1))
            bopt = np.zeros((nc,1))
            xidx = 0
            bidx = 0
            for v in m.getVars():
                if(xidx < ne):
                    xopt[xidx] = v.x
                    xidx+=1
                else:
                    bopt[bidx] = v.x
                    bidx+=1
            
            newC = interpret_milp_output(xopt,e)
            # print(newC) 
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
    
    return newC, timed_out, feas

# Interpreting MILP output:
def interpret_milp_output(x,e):
    lx = len(x)
    assert(lx == len(e))
    newC = [e[k] for k in range(lx) if x[k,0]==1]
    return newC

# Constructing the set of edges needed to be cut:
def construct_e(Pcut):
    n_cut = []
    e = [] # Edge vector
    path_indices = [] # Stores the indices of edges in e that belong to a path that needs to be cut
    # Parameters for paths that must be cut:
    for jj in range(len(Pcut)):
        Pcut_j = Pcut[jj]
        n_cut_j = len(Pcut_j) # No. of paths to cut from pj to pi+1
        path_indices_j = []    # Stores a list of edge_in_path for each path from pj to pi+1
        for path in Pcut_j:   # Finding edges in each path
            if path:
                edges_in_path = [] # Stores 
                for k in range(len(path)-1):
                    candidate_edge = (path[k], path[k+1]) # candidate edge to be cut
                    if candidate_edge not in e:
                        e.append(candidate_edge)
                    eidx = e.index(candidate_edge)
                    edges_in_path.append(eidx)
                path_indices_j.append(edges_in_path)
            else:
                n_cut_j -= 1
        path_indices.append(path_indices_j) # If there are no paths from pj to pi+1, this is an empty list
        n_cut.append(n_cut_j)
    return e, n_cut, path_indices

# Constructing A_cut:
def construct_cut_matrix(n_cut_paths, ne, cut_path_indices):
    A_cut = []
    for j in range(len(n_cut_paths)):
        cut_paths_j = cut_path_indices[j]
        row_idx = n_cut_paths[j]
        if row_idx > 0: # If there are any paths to be cut from pj to pi+1:
            A_cut_j = np.zeros((row_idx, ne))
            for jj in range(row_idx):
                cut_path_indices_j = cut_paths_j[jj]
                for eidx in cut_path_indices_j:
                    A_cut_j[jj, eidx] = 1
            A_cut.append(A_cut_j)
    return A_cut

# Constructing A_keep, D_keep and n_keep:
def construct_keep_parameters(Pkeep, MC_keep,e, ne):
    # Parameters relating to paths that must be kept (as many as possible):
    n_keep = []
    jidx = 0
    A_keep = []
    D_keep = []
    for Pkeep_j in Pkeep:
        n_keep_j = len(Pkeep_j)
        MCj = MC_keep[jidx]
        A_keep_j= None
        
        for path in Pkeep_j:
            if path:
                row = np.zeros((1,ne)) # Adding a row to A_jj+1
                for k in range(len(path)-1):
                    candidate_edge = (path[k], path[k+1])
                    if candidate_edge in e and candidate_edge in MCj: # If this is an edge that is a candidate to be cut in Pkeep and is a minimum cut edge from pj to pj+1:
                        eidx = e.index(candidate_edge)
                        row[0, eidx] = 1 
                if A_keep_j is None:
                    A_keep_j = row.copy()
                else:
                    A_keep_j = np.vstack((A_keep_j, row))
            else:
                n_keep_j -= 1
                
        if (n_keep_j>0):
            n_keep.append(n_keep_j)
            A_keep.append(A_keep_j)
            assert(A_keep_j is not None)
                
            # Creating D_keep:
            ones = np.ones((ne, 1))
            # Check dimensions match up:
            D_keep_j = A_keep_j @ ones
            
            assert(len(A_keep_j[0]) == ne) 
            assert(len(D_keep_j) == len(A_keep_j))
            D_keep.append(D_keep_j)
        jidx += 1
        
    return A_keep, D_keep, n_keep

# construct MILP parameters:
def construct_milp_params(Pcut, Pkeep, MC_keep, n):
    e, n_cut, cut_path_indices = construct_e(Pcut) # Finding paths that need to be cut
    ne = len(e) # Number of possible edges that could be cut
    n_cut_paths = np.sum(np.array(n_cut))
    ncuts = 0
    for ii in cut_path_indices:
        ncuts += len(ii)
    assert(ncuts == n_cut_paths)  # Sanity check
    A_cut = construct_cut_matrix(n_cut, ne, cut_path_indices) # Constructing first class of constraints
    A_keep, D_keep, n_keep = construct_keep_parameters(Pkeep, MC_keep,e, ne)   
    # Remove any zeros from the n_cut vector:
    ncut_new = []
    for ci in n_cut:
        if ci != 0:
            ncut_new.append(ci)
    # Sanity check assertions:
    assert(len(A_cut)==len(ncut_new))
    assert(len(A_keep)==len(n_keep))
    cost = construct_c(n_keep)      # uncomment for original cost function
    # cost = corrected_construct_c(n_keep, Pkeep) # Uncomment for modified cost
    return A_cut, ncut_new, A_keep, D_keep, n_keep, cost, e

# construct MILP parameters with a small adjustment to the cost
def construct_milp_params_corrected(Pcut, Pkeep, MC_keep, n):
    e, n_cut, cut_path_indices = construct_e(Pcut) # Finding paths that need to be cut
    ne = len(e) # Number of possible edges that could be cut
    n_cut_paths = np.sum(np.array(n_cut))
    ncuts = 0
    for ii in cut_path_indices:
        ncuts += len(ii)
    assert(ncuts == n_cut_paths)  # Sanity check
    A_cut = construct_cut_matrix(n_cut, ne, cut_path_indices) # Constructing first class of constraints
    A_keep, D_keep, n_keep = construct_keep_parameters(Pkeep, MC_keep,e, ne)   
    # Remove any zeros from the n_cut vector:
    ncut_new = []
    for ci in n_cut:
        if ci != 0:
            ncut_new.append(ci)
    # Sanity check assertions:
    assert(len(A_cut)==len(ncut_new))
    assert(len(A_keep)==len(n_keep))
    cost = construct_c(n_keep)  
    return A_cut, ncut_new, A_keep, D_keep, n_keep, cost, e

# Creating the cost vector for the MILP based on no. of paths that need to be preserved:
def construct_c(n_keep):
    n_keep_paths = int(np.sum(np.array(n_keep)))
    cost = np.ones((n_keep_paths, 1))
    idx = 0
    for n_keep_j in n_keep:
        cost[idx:idx + n_keep_j, 0] = 1.0/n_keep_j
        idx = idx + n_keep_j
    return cost

# Corrected cost vector to account for cycles:
def corrected_construct_c(n_keep, Pkeep):
    n_keep_paths = int(np.sum(np.array(n_keep)))  # Total number of paths that need to be kept
    cost = np.ones((n_keep_paths, 1))             # Creating a cost vector 
    # Process nodes in each path:
    nodes_keep = []                               # Nodes constituting the keep augmented paths
    node_prop = []                                # Nodes representing propositions
    nodes_Pj_list = []                            # Keep track of list of all nodes
    for Pj in Pkeep:                              # Iterating over all paths in Pkeep
        s = Pj[0][0]                              # Starting node in Pj
        t = Pj[0][-1]                             # Ending node in Pj
        if s not in node_prop:                    # Checking if starting node is in node_prop
            node_prop.append(s)                   # Append in node_prop if it is already not there
        if t not in node_prop:                    
            node_prop.append(t)                   
        nodes_Pj = list(set().union(*Pj)) # All nodes in Pj   # Union of all nodes in Pj
        nodes_Pj_list.append(nodes_Pj) # Storing all nodes for every Pj    # Append all nodes to Pj_list
        nodes_keep = list(set().union(nodes_keep, nodes_Pj))      # List of all nodes to keep
    nodes_keep = [ii for ii in nodes_keep if ii not in node_prop] # Removing nodes that are a part of node_prop
    node_count = dict()
    # Creating a dictionary of node counts:
    for nj in nodes_keep:
        node_count[nj] = 0                       # counting the number of times a node appears in the Pkeep list
    for nodes_Pj in nodes_Pj_list:               # Nodes in every Pj_list
        for nj in nodes_Pj:                      # For nj in the nodes_Pj
            if nj not in node_prop:              # Checking if nj is in node_prop
                node_count[nj] += 1              # Counting the number of nodes
    
    # Creating the cost vector:
    idx= 0
    for jj in range(len(Pkeep)):                 # The number of Pj paths in Pkeep
        njj = n_keep[jj]                         # Number of paths to keep in Pj
        Pj = Pkeep[jj] # jth pat                 # All paths from pj to pj+!
        for path in Pj:                          
            node_count_path = [node_count[nj] for nj in path[1:-1]]      # Tracking the node counts of nodes in a given path
            if node_count_path!=[]:
                max_node_share = max(node_count_path)                        # Finding the maximum
            else:
                max_node_share = 1
            cost[idx] = 1.0/njj * 1.0/max_node_share                         # decreasing the cost so that cycle paths are not weighted 
                                                                             # disproportionately higher than augmented paths with no cycle weights
            idx+=1
    assert(idx == n_keep_paths)
    return cost

# Finding nodes closer to goal than p2
# This function might have issues; written fast.
def sdistance(G, p2, g):
    Gc = G.copy()
    dp2 = len(nx.shortest_path(Gc, source=p2[0], target=g[0]))
    ancestor_tree = nx.dfs_tree(Gc.reverse(), source=g[0], depth_limit=dp2-1).reverse() # Finding all upstream nodes that could lead to G0 but not via the aug path nodes
    nodes_closer_goal = list(ancestor_tree)
    return nodes_closer_goal

# Returns all augmented paths from source s to sink t on graph G
def augment_paths(G, start, end, unit_capacities=False):
    s = start[0] # Assuming only one state for each source and sink
    t = end[0]
    P = []
 
    # Reading augmented paths from the flow network:
    Gc = G.copy()
    R, P = get_augmenting_path(Gc,s,t)
    
    Rf = R.graph["flow_value"]
    if unit_capacities:
        assert(len(P)==Rf) # number of augmenting paths should be equal to the flow value for unit capacities
    return P

# finding all augmnting paths p1, p2, ..., pn on final graph:
def all_augment_paths(G, props):
    Paug = []
    for ii in range(len(props)-1):
        Pi = augment_paths(G, props[ii], props[ii+1], unit_capacities=True)
        Paug.append(Pi)
    return Paug

# Function that prepares the min-cut edges graph:
# G is the original graph on which Pkeep_j (aug paths from j to j+1) and Pkeep (aug paths from k-1 to k) for all k
# are found. Function prepare_Gj constructs a graph Gj in which for each augmented path p in Pj, minimum-cut edges are
# computed on the graph Gp, which is the graph for which all nodes on augmented paths from Pk-1 to Pk are removed except for those on path p
def prepare_Gj(G, props, jj, Pkeep_j, Pkeep):
    MCj = []
    exclude_nodes = []
    for Pkeep_k in Pkeep:         # All aug paths from pk to pk+1
        nodes_k = list(set().union(*Pkeep_k)) # Collecting all nodes that are in Pkeep_k
        exclude_nodes = add_elems(exclude_nodes, nodes_k) # We don't want to exclude the current node pj from the final list
    for ap in Pkeep_j:
        node_remove_Gp = [node for node in exclude_nodes if node not in ap]
        Gp = G.copy()
        Gp.remove_nodes_from(node_remove_Gp)
        MCp = min_cut_edges(Gp, props[jj], props[jj+1])
        MCj = add_elems(MCj, MCp)
    return MCj

# Helper function to the prepare_Gj function:
# Adding elements(edges/nodes) from listB to listA that are not already in listA:
def add_elems(listA, listB):
    newA = listA.copy()
    for elem in listB:
        if elem not in listA:
            newA.append(elem)
    return newA

# Finds all paths in a graph 
# Finds minimum cut edges in a graph:
def min_cut_edges(G, start, end):
    MC = []
    s = start[0] # Assuming only one state for each source and sink
    t = end[0]
    R = edmonds_karp(G, s, t)
    fstar = R.graph["flow_value"]
    
    P = augment_paths(G,start,end)
    
    # Going over edges in the augmented paths
    for pathi in P: # Iterating over paths
        for ii in range(len(pathi)-1): # Iterating over edges in each path
            candidate_edge = (pathi[ii], pathi[ii+1])
            Gtemp = G.copy()
            # Snaity check:
            assert(candidate_edge in Gtemp.edges)
            Gtemp.remove_edge(*candidate_edge)
            Gtemp_residual = edmonds_karp(Gtemp, s, t)
            flow = Gtemp_residual.graph["flow_value"]
            if (fstar-flow)>= 1:             # If there is a decrease in flow caused by removal of this edge, then it is a minimum cut edge
                MC.append(candidate_edge)
    return MC


### Efficient versions of the same original functions above. Ex: If pj to pi+1 is a cut path, then a cu
# t path from pk (<j) to pi+1 is not included if it contains pj along the way.
# The only function need to be modified is cut_aug_paths(?)
def cut_aug_paths_eff(G, props, i):
    Pcut = []
    n_paths_to_cut = []
    for j in range(i-1, -1, -1): # Decreasing order from i-1 to 1
        n_paths_to_cut_j = 0
        Pcut_j = augment_paths(G, props[j], props[i+1]) # From pj to pi+1; which is index by pi
        Pcut_j_new = Pcut_j.copy()
        if (j < i-1): # checking if any of pj+1 to pi-1 are on aug path; if so, remove them
            higher_props = [props[k][0] for k in range(j+1, i)]   
            for Pj in Pcut_j:
                higher_prop_present = any(ph in Pj for ph in higher_props)
                if higher_prop_present:
                    Pcut_j_new.remove(Pj)
        if Pcut_j_new:
            n_paths_to_cut_j+=len(Pcut_j_new) # If there are paths to cut, update n_paths_to_cut
        Pcut.append(Pcut_j_new)
        n_paths_to_cut.append(n_paths_to_cut_j)
    Pcut_new = list(reversed(Pcut))
    return Pcut_new, n_paths_to_cut

