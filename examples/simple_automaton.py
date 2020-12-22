#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:51:09 2020

@author: apurvabadithela
"""
import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
import sys
sys.path.append('../')

from src.milp_functions import augment_paths, min_cut_edges
from src.restrict_transitions_cycles import remove_edges_corrected_cycle, remove_edges
from src.simulation_helpers import post_process_cuts, process_history

# Constructing graph induced by automaton:
G = nx.DiGraph()
G.add_edge("n1", "n2", capacity=1.0)
G.add_edge("n2", "n3", capacity=1.0)
G.add_edge("n2", "n4", capacity=1.0)
G.add_edge("n2", "n5", capacity=1.0)
G.add_edge("n3", "n6", capacity=1.0)
G.add_edge("n4", "n6", capacity=1.0)
G.add_edge("n5", "n6", capacity=1.0)
G.add_edge("n6", "n7", capacity=1.0)
G.add_edge("n2", "n8", capacity=1.0)
G.add_edge("n8", "n5", capacity=1.0)

# Specifying the goal and propositions:
g ="n7"
w = "n3"
q0 = "n1"

P = augment_paths(G, [q0], [g])
MC = min_cut_edges(G, [q0], [g])

props = [[q0], [w], [g]]
C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main, skip_iter = remove_edges(G, props, props[0], "ALL")

C = post_process_cuts(C, G)
Chist = process_history(Chist,G)
print("List of constrained edges: ")
print(C)
print("List of constrained edges in each iteration: ")
print(Chist)



