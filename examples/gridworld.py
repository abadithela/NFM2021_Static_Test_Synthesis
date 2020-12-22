#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:32:59 2020

@author: apurvabadithela
"""
# Generates plots for large number of examples:
# Use: restrict_transitions_complex
import networkx as nx
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from src.grid_functions import construct_grid, plot_static_map, base_grid, plot_final_map, plot_augment_paths
from src.simulation_helpers import run_iterations, generate_valid_config, post_process_cuts, process_history, to_directed
from src.milp_functions import all_augment_paths
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import matplotlib.animation as animation
from src.restrict_transitions_cycles import remove_edges_corrected_cycle, remove_edges

# Parameters:
# Takes as input the example number
def gridworld(ex, key):
    if ex == 1:
        M = 3 # Number of rows
        N = 3 # Number of columns
        g = "n"+str(M*N) # Goal state
        n = 2 # Number of propositions
        props = [["n1"],["n7"],["n3"]] # Propositions: [[p1], [p2]]
        props.append([g])
    
    if ex == 2:
        M = 4 # Number of rows
        N = 4 # Number of columns
        g = "n"+str(M*N) # Goal state
        n = 2 # Number of propositions
        props = [["n11"], ["n6"], ["n16"]] # Propositions: [[p1], [p2]]
        # props.append([g])
    
        
    # Constructing graphs:
    G = nx.DiGraph()
    nodes = ["n"+str(ii+1) for ii in range(M*N)]
    G.add_nodes_from(nodes)
    allow_diag = False # True means that diagonal transitions are allowed
    G = construct_grid(G, M, N, allow_diag)
    Gk = construct_grid(G, M, N, allow_diag)
    
    print(props)
    
    # Find test graph:
    t = time.time()
    
    
    C, Q0, Chist, discard, nkeep, niterations, nf_iterations, alg_fail_main, skip_iter = remove_edges(G, props, props[0], key)
    
    elapsed = time.time() - t
    
    # Plot augmenting paths:
    Paug = all_augment_paths(G, props[0:2])
    fig_aug, ax_aug = plot_augment_paths(Paug, G, M, N, props, Q0, [])
    
    if alg_fail_main==False:
        print(C)
        postC = post_process_cuts(C, G)
        print(postC)
        postChist = process_history(Chist, G)
        # Plotting figures:
        fig, ax = base_grid(M,N, props)
        FIG, AX, movie = plot_static_map(G, M, N, props, Q0, postChist)
        if discard!="infeasible":
            writer = animation.FFMpegFileWriter(fps=1, metadata=dict(artist='bww'), bitrate=1800)
            movie.save('movie.mp4', writer=writer) # Save movie
        fig_final, ax_final = plot_final_map(G,M,N,props,Q0,postC)
        
        # Plotting with augmented paths on the final graph:
        G.remove_edges_from(postC)
        Paug = all_augment_paths(G, props)
        fig_aug, ax_aug = plot_augment_paths(Paug, G, M, N, props, Q0, postC)
        
    else:
        postC = []
        postChist = [[] for ii in range(len(props))]
    # post_Chist = process_history(Chist, Gdir)
    
    print("Time taken: ")
    print(elapsed)

if __name__ == '__main__':
    print("Running 3-by-3 example by considering all shortest augmenting paths:")
    gridworld(1, "SAP")
    
    print("Running 4-by-4 example by considering all augmenting paths:")
    gridworld(2, "ALL")
    
    plt.show()