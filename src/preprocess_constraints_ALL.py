#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:04:21 2020

@author: apurvabadithela
"""


# ============ Pre-process constraints ===================== #
# Apurva Badithela: To pre-process flow constraints for the optimization algorithm
# ========================================================== #

import numpy as np
import time
import ipdb
import random
from grid_functions import construct_grid
import networkx as nx
import gurobipy as gp
import scipy.sparse as sp
from gurobipy import GRB
from networkx.algorithms.flow import shortest_augmenting_path, edmonds_karp
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from aug_path import get_augmenting_path
from networkx.algorithms.flow.utils import build_residual_network
from networkx.algorithms.shortest_paths import all_shortest_paths
from networkx.algorithms.simple_paths import all_simple_paths
# Toggle between all shortest paths and all simple paths

# Script to generate sets of all shortest augmenting paths from source s to sink t in a given graph.
# General data structures to use to construct Trees:
class SAP_Tree():
    def __init__(self,root):
        self.root = root
        self.children = []    # Children of root
        self.Nodes = []
    def addNode(self,obj):
        self.children.append(obj)
    def getAllNodes(self):
        self.Nodes.append(self.root)
        for child in self.children:
            self.Nodes.append(child.data)
        for child in self.children:
            if child.getChildNodes(self.Nodes) != None:
                child.getChildNodes(self.Nodes)
        print(*self.Nodes, sep = "\n")
        print('Tree Size:' + str(len(self.Nodes)))
    def getRoot_to_Leaf_Paths(self):
        SAP_sets = []
        shortest_paths = self.children
        for p1 in shortest_paths:
            P1 = list(all_paths(p1))
            SAP_sets.extend(P1)
        return SAP_sets
        
class Node():
    def __init__(self, data):
        self.data = data
        self.children = []
    def addNode(self,obj):
        self.children.append(obj)
    def getChildNodes(self,SAP_Tree):
        for child in self.children:
            if child.children:
                child.getChildNodes(SAP_Tree)
                SAP_Tree.append(child.data)
            else:
                SAP_Tree.append(child.data)
    def getData(self):
        return self.data
    
# finding all paths:
# Printing all paths:
def all_paths(node):
    if node is None:
        return 
    val = node
    children = node.children
    if any(children):
        for child in children: 
            for path in all_paths(child):
                yield [val.getData()] + path
    else:
        yield [val.getData()]

# Remove edges:
def remove_edges(G, P):
    Gnew = G.copy()
    for ii in range(1,len(P)):
        ei = (P[ii-1], P[ii])
        Gnew.remove_edge(*ei)
    return Gnew

# Given a list of set of augmented paths
def SAP_combinations(SAP):
    SAP_small = [] # Stores the pared down list of combinations
    for ii in range(len(SAP)):
        SAP_ii = SAP[ii]
        SAP_ii_stored = False
        for SAP_checked in SAP_small:
            is_SAP_ii_stored = all([elem in SAP_ii for elem in SAP_checked])
            if is_SAP_ii_stored:
                SAP_ii_stored = True
                break
        if not SAP_ii_stored:
            SAP_small.append(SAP_ii)
    return SAP_small

# Function to recursively add children:
def add_children(node, G, s, t):
    # Gnew = G.copy()
    Gnew = remove_edges(G, node)
    if nx.has_path(Gnew, s, t):
        node_ch = list(nx.all_simple_paths(Gnew, s, t))
    else:
        node_ch = []
    return node_ch, Gnew

# Finding all sets of shortest augmented paths:
def ALL_sets(G, s, t):
    root = Node([])
    t1 = SAP_Tree(root)
    saps_to_add = list(nx.all_simple_paths(G,s,t))
    sap_nodes_to_add = []
    G_list = [G.copy() for sch in saps_to_add]
    for ii in range(len(saps_to_add)):
        sap = saps_to_add[ii]
        sap_node = Node(sap)
        sap_nodes_to_add.append(sap_node)
    
    #     t1.addNode(sap_node)
    flag = 1 # Adding children of root node
    while saps_to_add!= []:
        #print(saps_to_add)
        new_saps_to_add = []
        new_sap_nodes_to_add = []
        G_list_new = []
        for ii in range(len(saps_to_add)):
            sap = saps_to_add[ii]
            G_sap = G_list[ii] # Finding the right Gsap
            sap_node = sap_nodes_to_add[ii]
            if flag==1:
                root.addNode(sap_node)
                t1.addNode(sap_node)
                if ii == len(saps_to_add)-1: # All the first set of nodes have been added
                    flag = 0
            sap_node_children, G_sap_edges_rem = add_children(sap, G_sap, s, t)
            for ch in sap_node_children:
                new_saps_to_add.append(ch)
                ch_node = Node(ch)
                new_sap_nodes_to_add.append(ch_node)
                sap_node.addNode(ch_node)  # add's child node
                G_list_new.append(G_sap_edges_rem)
                
                # Add as children to previous node:
                
        G_list = G_list_new.copy()       
        saps_to_add = new_saps_to_add.copy()
        sap_nodes_to_add = new_sap_nodes_to_add.copy()
    SAP = t1.getRoot_to_Leaf_Paths() # Returns permutations; we only want combinations
    SAP = SAP_combinations(SAP)
    return SAP

# Example:
G = nx.DiGraph()     # Difficult for a fully connected graph
for ii in range(7):
    G.add_node("n"+str(ii))
    G.add_edge("n0", "n1")
    G.add_edge("n0", "n2")
    G.add_edge("n1", "n3")
    G.add_edge("n2", "n3")
    G.add_edge("n3", "n4")
    G.add_edge("n3", "n5")
    G.add_edge("n3", "n7")
    G.add_edge("n4", "n6")
    G.add_edge("n5", "n6")
    G.add_edge("n7", "n6")
    G.add_edge("n0", "n7")

G1 = nx.DiGraph()
for ii in range(1,5):
    G1.add_node("n"+str(ii))
for ii in range(1,5):
    for jj in range(1,5):
        if ii != jj:
            G1.add_edge("n"+str(ii), "n"+str(jj))
G1.add_edge("n3", "n5")
# G1.add_edge("n1", "n5")
G1.add_edge("n5", "n4")
G1.remove_edge("n3", "n4")


G2 = nx.DiGraph()
G2.add_edge("n0", "n2")
G2.add_edge("n2", "n1")
G2.add_edge("n0", "n3")
G2.add_edge("n3", "n4")
G2.add_edge("n4", "n1")
G2.add_edge("n0", "n1")
# SAP= SAP_sets(G2, "n0", "n1")
# print(" ")
# print(SAP)

