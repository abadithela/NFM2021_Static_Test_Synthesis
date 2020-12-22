#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:01:37 2020

@author: apurvabadithelSETT"""
import networkx as nx
import time
import pickle as pkl
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
import sys
sys.path.append('../')

from src.grid_functions import construct_grid, plot_static_map, base_grid, plot_final_map, plot_augment_paths
from src.simulation_helpers import run_iterations, run_iterations_SAPs, run_iterations_random_graph
from time import gmtime, strftime
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
#===============================================================================================# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot Figures ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#===============================================================================================#
# Plotting Figures:
def plot_parametrized_figure(KEY, t, P, no_diag_data, diag_data=None):
    fig, ax = plt.subplots()
    RUNTIME = no_diag_data[0]
    RUNTIME_STD = no_diag_data[1]
    TIME_OUT = no_diag_data[2]
    ITER = no_diag_data[3]
    ILP_ITER = no_diag_data[4]
    Pk = P[0] # largest Pk
    color=iter(cm.rainbow(np.linspace(0,1,len(Pk)))) # Colors
    color = ['b', 'g']
    max_y = 0
    max_yerr = 0
    xlabels = [str(tuple(ti)) for ti in t]
    transform = [-0.05, 0, 0.05]
    for jj in range(len(Pk)):
        y = []
        x = []
        xticks = []
        space = transform[jj]
        yerr= []
        c = color[jj]
        for ii in range(len(t)):
            R_ii = RUNTIME[ii]
            Rt = RUNTIME_STD[ii]
            Tout_ii = TIME_OUT[ii]
            N_ii = ITER[ii]
            N_ilp_ii = ILP_ITER[ii]
            Pii = P[ii]
            assert(len(R_ii) == len(Tout_ii))
            if jj < len(R_ii):
                x.append(ii+1 + space)
                xticks.append(ii+1)
                y.append(R_ii[jj])
                yerr.append(Rt[jj])
        if max(y) > max_y:
            max_y = max(y)
        if max(yerr)> max_yerr:
            max_yerr = max(yerr)
        yerr= 0.434*np.array(yerr/y)
        plt.errorbar(x, y, yerr=yerr, c = c, alpha=.75, fmt=':', capsize=3, capthick=1, ls ='-', marker='o', linewidth=2, markersize=3, label=r'$|P| = $%s' % str(Pk[jj])) # total no. of props  (including goal)
        # plt.errorbar(x,y,yerr=yerr)
        ax.set_xticks(xticks)
        ax.set_xticklabels(tuple(xlabels))
    plt.yscale('log', nonposy='clip')
    # ax.set_yticks(np.arange(0, int(max_y+max_yerr)), (max_y+max_yerr)//1)
    if diag_data:
        RUNTIME = diag_data[0]
        RUNTIME_STD = diag_data[1]
        TIME_OUT = diag_data[2]
        ITER = diag_data[3]
        ILP_ITER = diag_data[4]
        Pk = P[0] # largest Pk
        color=iter(cm.rainbow(np.linspace(0,1,2*len(Pk)))) # Colors
        max_y = 0
        max_yerr=0
        for jj in range(len(Pk)):
            y = []
            x = []
            xticks = []
            space = transform[jj]
            yerr = []
            c = 'o--'+str(next(color))
            for ii in range(len(t)):
                R_ii = RUNTIME[ii]
                Rt = RUNTIME_STD[ii]
                Tout_ii = TIME_OUT[ii]
                N_ii = ITER[ii]
                N_ilp_ii = ILP_ITER[ii]
                Pii = P[ii]
                assert(len(R_ii) == len(Tout_ii))
                if jj < len(R_ii):
                    x.append(ii+1 + space)
                    xticks.append(ii+1)
                    y.append(R_ii[jj])
                    yerr.append(Rt[jj])
            if max(y) > max_y:
                max_y = max(y)
            if max(yerr)> max_yerr:
                max_yerr = max(yerr)
            plt.errorbar(x, y, yerr = yerr, c =c, linewidth=2, markersize=2, label=r'diag: $|P| = $%s' % str(Pk[jj]+1))
            #ax.plot(x, y, yerr=yerr)
    plt.legend()
    ax.set(xlabel='Grid size (M,N)', ylabel='Runtime (s)')
    # fig.savefig(KEY+".csv")
    fig.savefig(KEY+".png", dpi=300)
    return fig, ax 

# Run parametrized figure for square grids:
def plot_parametrized_figure_square_grids(KEY, t, P, no_diag_data, diag_data=None):
    fig, ax = plt.subplots()
    RUNTIME = no_diag_data[0]
    RUNTIME_STD = no_diag_data[1]
    TIME_OUT = no_diag_data[2]
    ITER = no_diag_data[3]
    ILP_ITER = no_diag_data[4]
    Pk = P[0] # largest Pk
    color=iter(cm.rainbow(np.linspace(0,1,len(Pk)))) # Colors
    color = ['b', 'g', 'm']
    max_y = 0
    max_yerr = 0
    xlabels = t
    transform = [-0.05, 0, 0.05]
    for jj in range(len(Pk)):
        y = []
        x = []
        yerr=[]
        xticks = []
        space = transform[jj]
        c = color[jj]
        for ii in range(len(t)):
            R_ii = RUNTIME[ii]
            Tout_ii = TIME_OUT[ii]
            N_ii = ITER[ii]
            N_ilp_ii = ILP_ITER[ii]
            Pii = P[ii]
            Rt = RUNTIME_STD[ii]
            assert(len(R_ii) == len(Tout_ii))
            if jj < len(R_ii):
                x.append(ii+1 + space)
                xticks.append(ii+1)
                y.append(R_ii[jj])
                yerr.append(Rt[jj])
        if max(y) > max_y:
            max_y = max(y)
        if max(yerr)> max_yerr:
            max_yerr = max(yerr)
        plt.errorbar(x, y, yerr = yerr, c = c, alpha=.75, fmt=':', capsize=3, capthick=1, ls ='-', marker='o', linewidth=2, markersize=3, label=r'$|P| = $%s' % str(Pk[jj])) # total no. of props  (including goal)
        # plt.errorbar(x,y,yerr=yerr)
        ax.set_xticks(xticks)
        ax.set_xticklabels(t)
    plt.yscale('log', nonposy='clip')
    # ax.set_yticks(np.arange(0, int(max_y+max_yerr)), (max_y+max_yerr)//1)
    if diag_data:
        RUNTIME = diag_data[0]
        RUNTIME_STD = diag_data[1]
        TIME_OUT = diag_data[2]
        ITER = diag_data[3]
        ILP_ITER = diag_data[4]
        Pk = P[0] # largest Pk
        color=iter(cm.rainbow(np.linspace(0,1,2*len(Pk)))) # Colors
        
        for jj in range(len(Pk)):
            y = []
            x = []
            yerr= []
            xticks = []
            space = transform[jj]
            c = 'o--'+str(next(color))
            for ii in range(len(t)):
                R_ii = RUNTIME[ii]
                Rt  = RUNTIME_STD[ii]
                Tout_ii = TIME_OUT[ii]
                N_ii = ITER[ii]
                N_ilp_ii = ILP_ITER[ii]
                Pii = P[ii]
                assert(len(R_ii) == len(Tout_ii))
                if jj < len(R_ii):
                    x.append(ii+1+space)
                    xticks.append(ii+1)
                    y.append(R_ii[jj])
                    yerr.append(Rt[jj])
            plt.errorbar(x, y, yerr = yerr, c = c, alpha=.75, fmt=':', capsize=3, capthick=1, ls ='-', marker='o', linewidth=2, markersize=3, label=r'$|P| = $%s' % str(Pk[jj]))
    plt.legend()
    ax.set(xlabel='Grid size (t)', ylabel='Runtime (s)')
    # fig.savefig(KEY+".csv")
    fig.savefig(KEY+".png")
    return fig, ax 

# Plotting Figures:
def plot_parametrized_figure_rg(KEY, M, N, P, no_diag_data, diag_data=None):
    fig, ax = plt.subplots()
    RUNTIME = no_diag_data[0]
    TIME_OUT = no_diag_data[1]
    ITER = no_diag_data[2]
    ILP_ITER = no_diag_data[3]
    Pk = P[0] # largest Pk
    color=iter(cm.rainbow(np.linspace(0,1,len(Pk)))) # Colors
    color = iter(['b', 'k'])
    for jj in range(len(Pk)):
        y = []
        x = []
        c = next(color)
        for ii in range(len(M)):
            R_ii = RUNTIME[ii]
            Tout_ii = TIME_OUT[ii]
            N_ii = ITER[ii]
            N_ilp_ii = ILP_ITER[ii]
            Pii = P[ii]
            assert(len(R_ii) == len(Tout_ii))
            if jj < len(R_ii):
                x.append(M[ii])
                y.append(R_ii[jj])
        ax.plot(x, y, c = c, ls ='-', marker='o', linewidth=2, markersize=2, label=r'$|P| = $%s' % str(Pk[jj]+1))
        ax.set_title(N+" edges")
        ax.set_xticks(M)
        ax.set_yticks(np.arange(0, int(max(y))+1, 0.5))
    if diag_data:
        RUNTIME = diag_data[0]
        TIME_OUT = diag_data[1]
        ITER = diag_data[2]
        ILP_ITER = diag_data[3]
        Pk = P[0] # largest Pk
        color=iter(cm.rainbow(np.linspace(0,1,2*len(Pk)))) # Colors
        for jj in range(len(Pk)):
            y = []
            x = []
            c = 'o--'+str(next(color))
            for ii in range(len(M)):
                R_ii = RUNTIME[ii]
                Tout_ii = TIME_OUT[ii]
                N_ii = ITER[ii]
                N_ilp_ii = ILP_ITER[ii]
                Pii = P[ii]
                assert(len(R_ii) == len(Tout_ii))
                if jj < len(R_ii):
                    x.append(M[ii])
                    y.append(R_ii[jj])
            ax.plot(x, y, c, linewidth=2, markersize=2, label=r'diag: $|P| = $%s' % str(Pk[jj]+1))
    plt.legend()
    ax.set(xlabel='Number of nodes (n)', ylabel='Runtime (s)')
    # fig.savefig(KEY+".csv")
    name = "Data/random_"+str(KEY)+"_"+str(N)+"_"
    figname = name + "#"+".png"
    fig.savefig(figname, dpi = 300)
    return fig, ax 

# Function to save data:
def save_data(KEY, t, P, no_diag_data, diag_data=None):
    current_directory = os.getcwd()
    name = "Data/New/#/param_grid_"+str(KEY)
    name = name.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    final_directory = os.path.join(current_directory, name)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    # =============== No diagonal data ============================== #
    ft = name + "_rows"+".dat"
    # ft = ft.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(t, open(ft,"wb"))
    # ft.close()
    
    fP = name + "_"+"P"+".dat"
    # fP = fP.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(P, open(fP,"wb"))
    # fP.close()
    
    ftime_avg = name + "_"+"time_avg"+".dat"
    # ftime_avg = ftime_avg.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[0], open(ftime_avg,"wb"))
    # ftime_avg.close()
    
    TIME_AVG_STD_DEV = name + "_"+"time_avg_std_dev"+".dat"
    # ftimed_out = ftimed_out.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[1], open(TIME_AVG_STD_DEV,"wb"))
    
    ftimed_out = name + "_"+"timed_out"+".dat"
    # ftimed_out = ftimed_out.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[2], open(ftimed_out,"wb"))
    # ftimed_out.close()
    
    ftotal_iter = name+ "_"+"total_iter"+".dat"
    # ftotal_iter = ftotal_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[3], open(ftotal_iter,"wb"))
    # ftotal_iter.close()
    
    fILP_iter = name+ "_"+"ILP"+".dat"
    # fILP_iter = fILP_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[4], open(fILP_iter,"wb"))
    # fILP_iter.close()
    
    # =============== Diagonal data ============================== #
    if diag_data:
        name = "Data/#/DIAG_param_grid_"+str(KEY)
        name = name.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        ft = name + "_"+"t"+".dat"
        # ft = ft.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(t, open(ft,"wb"))
        # ft.close()
        
        fP = name + "_"+"P"+".dat"
        # fP = fP.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(P, open(fP,"wb"))
        # fP.close()
        
        ftime_avg = name + "_"+"time_avg"+".dat"
        # ftime_avg = ftime_avg.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[0], open(ftime_avg,"wb"))
        # ftime_avg.close()
        
        TIME_AVG_STD_DEV = name + "_"+"time_avg_std_dev"+".dat"
        # ftimed_out = ftimed_out.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(no_diag_data[1], open(TIME_AVG_STD_DEV,"wb"))
        
        ftimed_out = name + "_"+"timed_out"+".dat"
        # ftimed_out = ftimed_out.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(no_diag_data[2], open(ftimed_out,"wb"))
    
        ftotal_iter = name+ "_"+"total_iter"+".dat"
        # ftotal_iter = ftotal_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[3], open(ftotal_iter,"wb"))
        # ftotal_iter.close()
        
        fILP_iter = name+ "_"+"ILP"+".dat"
        # fILP_iter = fILP_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[4], open(fILP_iter,"wb"))
        # fILP_iter.close()

# Function to save data for random graphs:
def save_data_random_graph(KEY, M, N, P, no_diag_data, diag_data=None):
    name = "Data/param_grid_"+str(KEY)+"_"+str(N)
    # =============== No diagonal data ============================== #
    ft = name + "_"+"grids"+"_#"+".dat"
    ft = ft.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(M, open(ft,"wb"))
    # ft.close()
    
    fP = name + "_"+"P"+"_#"+".dat"
    fP = fP.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(P, open(fP,"wb"))
    # fP.close()
    
    ftime_avg = name + "_"+"time_avg"+"_#"+".dat"
    ftime_avg = ftime_avg.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[0], open(ftime_avg,"wb"))
    # ftime_avg.close()
    
    ftimed_out = name + "_"+"timed_out"+"_#"+".dat"
    ftimed_out = ftimed_out.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[1], open(ftimed_out,"wb"))
    # ftimed_out.close()
    
    ftotal_iter = name+ "_"+"total_iter"+"_#"+".dat"
    ftotal_iter = ftotal_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[2], open(ftotal_iter,"wb"))
    # ftotal_iter.close()
    
    fILP_iter = name+ "_"+"ILP"+"_#"+".dat"
    fILP_iter = fILP_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
    pkl.dump(no_diag_data[3], open(fILP_iter,"wb"))
    # fILP_iter.close()
    
    # =============== Diagonal data ============================== #
    if diag_data:
        name = "param_grid_"+str(KEY)+"_"+str(N)
        ft = name + "_"+"M"+"_#"+".dat"
        ft = ft.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(M, open(ft,"wb"))
        # ft.close()
        
        fP = name + "_"+"P"+"_#"+".dat"
        fP = fP.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(P, open(fP,"wb"))
        # fP.close()
        
        ftime_avg = name + "_"+"time_avg"+"_#"+".dat"
        ftime_avg = ftime_avg.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[0], open(ftime_avg,"wb"))
        # ftime_avg.close()
        
        ftimed_out = name + "_"+"timed_out"+"_#"+".dat"
        ftimed_out = ftimed_out.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[1], open(ftimed_out,"wb"))
        # ftimed_out.close()
        
        ftotal_iter = name+ "_"+"total_iter"+"_#"+".dat"
        ftotal_iter = ftotal_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[2], open(ftotal_iter,"wb"))
        # ftotal_iter.close()
        
        fILP_iter = name+ "_"+"ILP"+"_#"+".dat"
        fILP_iter = fILP_iter.replace("#", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))
        pkl.dump(diag_data[3], open(fILP_iter,"wb"))
        # fILP_iter.close()
# ==============================================================================================#
# Parametrized gridworld iterative solver:
#===============================================================================================#
# Running SAP_augmented_paths constraints:
def run_SAPs(diag):
    Niter = 50 # No. of iterations for each example
    # t = [3,4,5,6,7, 8, 9] # Grid size
    t = [4,5,6,7,8] # Grid size
    lM = len(t)
    lN = len(t)
    P = [[3,4] for ii in range(len(t))]
    time_arr_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
    timed_out_num_no_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
    time_avg_std_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
    time_avg_std_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
    
    time_arr_diag = [[] for iM in range(lM)]  # Stores time taking to solve a problem
    timed_out_num_diag = [[] for iM in range(lM)]  # Stores no. of infeasible iterations
    fail_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    fail_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    total_iterations_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    total_iterations_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    total_iterations_ILP_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    total_iterations_ILP_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    
    KEY = "SAP" # Use only SAP constraints
    for ii in range(lM):
        iM = t[ii]
        iN = t[ii]
        nstates = iM*iN
        nprops = P[ii]
        nP = len(nprops)# No. of propositions
        time_arr_iM_iN_no_diag = [0 for inP in range(nP)]
        time_avg_std_iM_iN_no_diag = [0 for inP in range(nP)]
        timed_out_iM_iN_no_diag = [0 for inP in range(nP)]
        total_iteration_iM_iN_no_diag = [0 for inP in range(nP)] 
        total_ILP_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
        
        if diag:
            time_arr_iM_iN_diag = [0 for inP in range(nP)]
            time_avg_std_iM_iN_diag = [0 for inP in range(nP)]
            timed_out_iM_iN_diag = [0 for inP in range(nP)]
            fail_iM_iN_diag = [0 for inP in range(nP)]
            fail_iM_iN_no_diag = [0 for inP in range(nP)]
            total_iteration_iM_iN_diag = [0 for inP in range(nP)]
            total_ILP_iteration_iM_iN_diag = [0 for inP in range(nP)]
        
        
        for inP in range(len(nprops)):
            n_inP = nprops[inP]
            print("Computing data for t = "+str(iM)+" and nprops = "+str(n_inP))
            time_avg, timed_out_avg, total_iter_avg, total_ILP_avg, time_avg_std = run_iterations_SAPs(iM, iN, n_inP, Niter, False, KEY)  # Nodiagonal transitions
            time_avg_std_iM_iN_no_diag[inP-1] = time_avg_std
            
            time_arr_iM_iN_no_diag[inP-1] = time_avg
            timed_out_iM_iN_no_diag[inP-1] = timed_out_avg
            total_iteration_iM_iN_no_diag[inP-1] = total_iter_avg
            total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg
            
            if diag:
                time_avg, timed_out_avg, total_iter_avg, total_ILP_avg, time_avg_std = run_iterations_SAPs(iM, iN, inP, Niter, True, KEY)  # Nodiagonal transitions
                time_arr_iM_iN_diag[inP-1] = time_avg
                time_avg_std_iM_iN_diag[inP-1] = time_avg_std
                timed_out_iM_iN_diag[inP-1] = timed_out_avg
                total_iteration_iM_iN_diag[inP-1] = total_iter_avg  
                total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg
            
        time_arr_no_diag[ii] = time_arr_iM_iN_no_diag.copy()
        time_avg_std_no_diag[ii] = time_avg_std_iM_iN_no_diag.copy() 
        timed_out_num_no_diag[ii] = timed_out_iM_iN_no_diag.copy()
        total_iterations_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
        total_iterations_ILP_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
        
        if diag:
            time_arr_diag[ii] = time_arr_iM_iN_diag.copy()
            time_avg_std_diag[ii] = time_avg_std_iM_iN_diag.copy()
            timed_out_num_diag[ii] = timed_out_iM_iN_diag.copy()
            total_iterations_diag[ii] = total_iteration_iM_iN_diag.copy()
            total_iterations_ILP_diag[ii] = total_iteration_iM_iN_diag.copy()

    # no diagonals:
        if diag:
            save_data(KEY, t[ii], P, [time_arr_iM_iN_no_diag, time_avg_std_iM_iN_no_diag, timed_out_iM_iN_no_diag, total_iteration_iM_iN_no_diag, total_ILP_iteration_iM_iN_no_diag],[time_arr_iM_iN_diag, time_avg_std_iM_iN_diag, timed_out_iM_iN_diag, total_iteration_iM_iN_diag, total_ILP_iteration_iM_iN_diag])
        else:
            save_data(KEY, t[ii], P, [time_arr_iM_iN_no_diag, time_avg_std_iM_iN_no_diag, timed_out_iM_iN_no_diag, total_iteration_iM_iN_no_diag, total_ILP_iteration_iM_iN_no_diag]) # ,[time_arr_diag, time_avg_std_diag, timed_out_num_diag, total_iterations_diag, total_iterations_ILP_diag])
    if diag:
        fig, ax = plot_parametrized_figure_square_grids(KEY, t, P, [time_arr_no_diag, time_avg_std_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag],[time_arr_diag, time_avg_std_diag, timed_out_num_diag, total_iterations_diag, total_iterations_ILP_diag])
    else:
        fig, ax = plot_parametrized_figure_square_grids(KEY, t, P, [time_arr_no_diag, time_avg_std_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag]) #,[time_arr_diag, time_avg_std_diag, timed_out_num_diag, total_iterations_diag, total_iterations_ILP_diag])
    return fig, ax

# Running ALL_augmented_paths constraints:
def run_ALL(diag):
    Niter = 50 # No. of iterations for each example
    t = [[3,3],[3,4],[3,5],[4,4]] # Grid size
    
    # t = [[3,3],[3,4]]
    lM = len(t)
    lN = len(t)
    P = [[2,3], [2,3], [2,3], [2,3], [2,3]]

    
    time_arr_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
    timed_out_num_no_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
    time_avg_std_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
    time_avg_std_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
    
    time_arr_diag = [[] for iM in range(lM)]  # Stores time taking to solve a problem
    timed_out_num_diag = [[] for iM in range(lM)]  # Stores no. of infeasible iterations
    fail_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    fail_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    total_iterations_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    total_iterations_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    total_iterations_ILP_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    total_iterations_ILP_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    
    KEY = "ALL" # Use all simple path constraints:
    for ii in range(lM):
        iM = t[ii][0]
        iN = t[ii][1]
        nstates = iM*iN
        nprops = P[ii]
        nP = len(nprops)# No. of propositions
        
        time_arr_iM_iN_no_diag = [0 for inP in range(nP)]
        time_avg_std_iM_iN_no_diag = [0 for inP in range(nP)]
        timed_out_iM_iN_no_diag = [0 for inP in range(nP)]
        
        time_arr_iM_iN_diag = [0 for inP in range(nP)]
        time_avg_std_iM_iN_diag = [0 for inP in range(nP)]
        timed_out_iM_iN_diag = [0 for inP in range(nP)]
        fail_iM_iN_diag = [0 for inP in range(nP)]
        
        fail_iM_iN_no_diag = [0 for inP in range(nP)]
        total_iteration_iM_iN_diag = [0 for inP in range(nP)]
        total_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
        total_ILP_iteration_iM_iN_diag = [0 for inP in range(nP)]
        total_ILP_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
        
        for inP in range(len(nprops)):
            n_inP = nprops[inP]
            print("Computing data for rows = "+str(iM)+" and cols = "+str(iN)+"and nprops = "+str(n_inP))
            time_avg, timed_out_avg, total_iter_avg, total_ILP_avg, time_avg_std = run_iterations(iM, iN, n_inP, Niter, False, KEY)  # Nodiagonal transitions
            time_avg_std_iM_iN_no_diag[inP-1] = time_avg_std
            
            time_arr_iM_iN_no_diag[inP-1] = time_avg
            timed_out_iM_iN_no_diag[inP-1] = timed_out_avg
            total_iteration_iM_iN_no_diag[inP-1] = total_iter_avg
            total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg
            
            if diag:
                time_avg, timed_out_avg, total_iter_avg, total_ILP_avg, time_avg_std = run_iterations(iM, iN, inP, Niter, True, KEY)  # Nodiagonal transitions
                time_arr_iM_iN_diag[inP-1] = time_avg
                time_avg_std_iM_iN_diag[inP-1] = time_avg_std
                timed_out_iM_iN_diag[inP-1] = timed_out_avg
                total_iteration_iM_iN_diag[inP-1] = total_iter_avg  
                total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg

        time_arr_no_diag[ii] = time_arr_iM_iN_no_diag.copy()
        time_avg_std_no_diag[ii] = time_avg_std_iM_iN_no_diag.copy() 
        timed_out_num_no_diag[ii] = timed_out_iM_iN_no_diag.copy()

        total_iterations_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
        total_iterations_ILP_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
    
    # With diagonals:
    save_data(KEY, t, P, [time_arr_no_diag, time_avg_std_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag])
    fig, ax = plot_parametrized_figure(KEY, t, P, [time_arr_no_diag, time_avg_std_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag])
    
    return fig, ax

# ====================================================================================================
# Random Graphs
# ====================================================================================================
# Running SAP_augmented_paths constraints:
def run_rg_SAPs():
    Niter = 20 # No. of iterations for each example
    t = [3,4,5,6,7] # Grid size
    lM = len(t)
    lN = len(t)
    P = [[2,3,4], [2,3], [2,3], [2,3], [2,3]]
    time_arr_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
    timed_out_num_no_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
    time_arr_diag = [[] for iM in range(lM)]  # Stores time taking to solve a problem
    timed_out_num_diag = [[] for iM in range(lM)]  # Stores no. of infeasible iterations
    fail_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    fail_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    total_iterations_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    total_iterations_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    total_iterations_ILP_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
    total_iterations_ILP_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
    
    KEY = "SAP" # Use only SAP constraints
    for ii in range(lM):
        iM = t[ii]
        iN = t[ii]
        nstates = iM*iN
        nprops = P[ii]
        nP = len(nprops)# No. of propositions
        time_arr_iM_iN_no_diag = [0 for inP in range(nP)]
        timed_out_iM_iN_no_diag = [0 for inP in range(nP)]
        time_arr_iM_iN_diag = [0 for inP in range(nP)]
        timed_out_iM_iN_diag = [0 for inP in range(nP)]
        fail_iM_iN_diag = [0 for inP in range(nP)]
        fail_iM_iN_no_diag = [0 for inP in range(nP)]
        total_iteration_iM_iN_diag = [0 for inP in range(nP)]
        total_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
        total_ILP_iteration_iM_iN_diag = [0 for inP in range(nP)]
        total_ILP_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
        
        for inP in range(len(nprops)):
            n_inP = nprops[inP]
            print("Computing data for t = "+str(iM)+" and nprops = "+str(n_inP))
            time_avg, timed_out_avg, total_iter_avg, total_ILP_avg = run_iterations(iM, iN, n_inP, Niter, False, KEY)  # Nodiagonal transitions
            time_arr_iM_iN_no_diag[inP-1] = time_avg
            timed_out_iM_iN_no_diag[inP-1] = timed_out_avg
            total_iteration_iM_iN_no_diag[inP-1] = total_iter_avg
            total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg
            
            time_avg, timed_out_avg, total_iter_avg, total_ILP_avg = run_iterations(iM, iN, inP, Niter, True, KEY)  # Nodiagonal transitions
            time_arr_iM_iN_diag[inP-1] = time_avg
            timed_out_iM_iN_diag[inP-1] = timed_out_avg
            total_iteration_iM_iN_diag[inP-1] = total_iter_avg  
            total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg
            
        time_arr_no_diag[ii] = time_arr_iM_iN_no_diag.copy()
        timed_out_num_no_diag[ii] = timed_out_iM_iN_no_diag.copy()
        time_arr_diag[ii] = time_arr_iM_iN_diag.copy()
        timed_out_num_diag[ii] = timed_out_iM_iN_diag.copy()
        total_iterations_diag[ii] = total_iteration_iM_iN_diag.copy()
        total_iterations_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
        total_iterations_ILP_diag[ii] = total_iteration_iM_iN_diag.copy()
        total_iterations_ILP_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
    
    # no diagonals:
    save_data(KEY, t, P, [time_arr_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag], [time_arr_diag, timed_out_num_diag, total_iterations_diag, total_iterations_ILP_diag])
    fig, ax = plot_parametrized_figure(KEY, t, P, [time_arr_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag], [time_arr_diag, timed_out_num_diag, total_iterations_diag, total_iterations_ILP_diag])
    FIG = []
    AX = []
    return FIG, AX

# Running ALL_augmented_paths constraints:
def run_rg_ALL():
    FIG = []
    AX = []
    Niter = 2 # No. of iterations for each example
    M = [9,10,11,12,13,14] # Number of nodes
    N = [lambda M: 2*M, lambda M: 3*M]#, lambda M: int(np.floor(M*(M-1)/4))] # No. of edges
    lM = len(M)
    formula = ["2N", "3N"]
    # lN = len(t)
    for ilN in range(len(N)):
        N_expr = N[ilN]
        flN = formula[ilN]# Formula
        P = [[2,3], [2,3], [2,3], [2,3], [2,3], [2,3], [2], [2]] # For each N, how many props for every graph
        time_arr_no_diag = [[] for iM in range(lM)] # Stores time taking to solve a problem
        timed_out_num_no_diag = [[] for iM in range(lM)]# Stores no. of infeasible iterations
        time_arr_diag = [[] for iM in range(lM)]  # Stores time taking to solve a problem
        timed_out_num_diag = [[] for iM in range(lM)]  # Stores no. of infeasible iterations
        fail_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
        fail_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
        total_iterations_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
        total_iterations_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
        total_iterations_ILP_no_diag = [[] for iM in range(lM)] # iterations when no diagonal transitions are possible
        total_iterations_ILP_diag = [[] for iM in range(lM)] # Iterations when diagonal transitions are possible
        KEY = "ALL" # Use all simple path constraints:
        for ii in range(lM):
            iM = M[ii]
            iN = N_expr(M[ii])
            nstates = iM*iN
            nprops = P[ii]
            nP = len(nprops)# No. of propositions
            time_arr_iM_iN_no_diag = [0 for inP in range(nP)]
            timed_out_iM_iN_no_diag = [0 for inP in range(nP)]
            total_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
            total_ILP_iteration_iM_iN_no_diag = [0 for inP in range(nP)]
            
            for inP in range(len(nprops)):
                n_inP = nprops[inP]
                print("Computing data for t = "+str(iM)+" and nprops = "+str(n_inP))
                time_avg, timed_out_avg, total_iter_avg, total_ILP_avg, time_avg_std = run_iterations_random_graph(iM, iN, n_inP, Niter, False, KEY)  # Nodiagonal transitions
                time_arr_iM_iN_no_diag[inP-1] = time_avg
                timed_out_iM_iN_no_diag[inP-1] = timed_out_avg
                total_iteration_iM_iN_no_diag[inP-1] = total_iter_avg
                total_ILP_iteration_iM_iN_no_diag[inP-1] = total_ILP_avg
            
            time_arr_no_diag[ii] = time_arr_iM_iN_no_diag.copy()
            timed_out_num_no_diag[ii] = timed_out_iM_iN_no_diag.copy()
            total_iterations_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()
            total_iterations_ILP_no_diag[ii] = total_iteration_iM_iN_no_diag.copy()   
        
        # With diagonals:
        save_data_random_graph(KEY, M, flN, P, [time_arr_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag])
        fig, ax = plot_parametrized_figure_rg(KEY, M, flN, P, [time_arr_no_diag, timed_out_num_no_diag, total_iterations_no_diag, total_iterations_ILP_no_diag])
        FIG.append(fig)
        AX.append(ax)
    return FIG, AX

if __name__ == '__main__':
    print("Running all_augmenting paths simulations for random gridworlds")
    fig, ax = run_ALL(False)
    
    print("Running shortest_augmenting_paths simulations for random gridworlds")
    fig2, ax2 = run_SAPs(False)
    
    plt.show()