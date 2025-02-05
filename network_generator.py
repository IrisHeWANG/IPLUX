import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_graph(num_of_nodes,num_of_edges):
    '''
        1）Assign four roles to each node:
            - sparse_eq
            - sparse_in
            - both
            - neither
        2) A: appropriate adj matrix, based on sparse sets.
        3) Randomly add links to current graph generated by A, until connected.
        4) Optioal: plot 
            - general plot.
            - plot with roles.
        
    '''
    G=nx.gnm_random_graph(num_of_nodes,num_of_edges,seed=0)
    Adj = adjacency(G)
    print(Adj)
    sparse = {}
    for i in range(num_of_nodes):
        sparse[i]=[i]
        for j in range(num_of_nodes):
            if Adj[i][j] == 1:
                sparse[i].append(j)

    sparse_eq = sparse
    sparse_in = sparse

    

    k=0
    while nx.is_connected(G)==False:
        G=nx.gnm_random_graph(num_of_nodes,num_of_edges,seed=0)
        k+=1
    print("Check if connected: ", nx.is_connected(G))

    return G,Adj,sparse_eq,sparse_in

def adjacency(G):
    '''
        Generate the adjacency_matrix of graph G.
    '''
    A=nx.adjacency_matrix(G).A
    return A

