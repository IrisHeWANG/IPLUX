import pandas as pd 
import numpy as np
import cvxpy as cp
import math
import csv

def check_linear_independent(Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    A_all = np.zeros((m_global+m_sparse*num_of_nodes,dim*num_of_nodes))
    for i in range(num_of_nodes):
        A_all[0:m_global,i*dim:(i+1)*dim] = Data[i]['A']
        if Data[i]['A_i'] != {}:
            eq_set_i = Data[i]['A_i'].keys()
            for j in eq_set_i:
                A_all[m_global + j*m_sparse:m_global + (j+1)*m_sparse, i*dim:(i+1)*dim] = Data[i]['A_i'][j]

    if np.linalg.matrix_rank(A_all) == (m_global + m_sparse*num_of_nodes):
        print("Check Independent grad_h: Pass.")
        
    return
def eigval_max(A):
    return max(np.linalg.eigvals(A))

def A_s(Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    A_s_all = np.zeros((m_sparse*num_of_nodes,dim*num_of_nodes))
    for i in range(num_of_nodes):
        num_neighbors = len(Data[i]['sparse_eq'])
        for j in range(num_neighbors):
            j_inx = Data[i]['sparse_eq'][j]
            A_s_all[m_sparse*i:m_sparse*(i+1),dim*j_inx:dim*(j_inx+1)] = Data[i]['A_j'][j]
    return A_s_all

################################################################################
                           # IPLUX Part         
################################################################################

def metropolis(adjacency_matrix):
    num_of_nodes = adjacency_matrix.shape[0]
    metropolis=np.zeros((num_of_nodes,num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if adjacency_matrix[i,j]==1:
                d_i = np.sum(adjacency_matrix[i,:])
                d_j = np.sum(adjacency_matrix[j,:])
                metropolis[i,j]=1/(1+max(d_i,d_j))
        metropolis[i,i]=1-sum(metropolis[i,:])
    return metropolis

def L_g(Data_i,dim):
    x = cp.Variable(dim)
    a_i = Data_i['a']
    c_i = Data_i['c']

    aa_i = Data_i['aa']
    cc_i = Data_i['cc']

    prob = cp.norm(2*(x-aa_i))
    constraint = [cp.norm(x-a_i)**2 <= c_i]
    prob = cp.Problem(cp.Minimize(prob),constraint)
    prob.solve()

    return prob.value

def L_g_s(Data_i,dim):
    if Data_i['sparse_in'] == []:
        return 0
    
    a_i = Data_i['a']
    c_i = Data_i['c']
    max_g = 0
    for j in range(len(Data_i['sparse_in'])):
        a_ij = Data_i['a_j'][j]

        x = cp.Variable(dim)
        prob = cp.norm(2*(x-a_ij))

        constraint = [cp.norm(x-a_i)**2 <= c_i]
        prob = cp.Problem(cp.Minimize(prob),constraint)
        prob.solve()
        if prob.value > max_g:
            max_g = prob.value

    return max_g
    

def alpha_lower(Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    max_sum_S_j_in = 0
    max_L_f = 0
    max_L_g = 0
    max_L_g_s = 0
    for i in range(num_of_nodes):
        P_i = Data[i]['P']
        L_f = eigval_max(2*P_i) 
        if L_f >= max_L_f:
            max_L_f = L_f
        L_g_i = L_g(Data[i],dim)
        if L_g_i > max_L_g:
            max_L_g = L_g_i
        L_g_s_i = L_g_s(Data[i],dim)
        if L_g_s_i > max_L_g_s:
            max_L_g_s = L_g_s_i

        sum_S_j_in = 3*len(Data[i]['a_i'])
        if sum_S_j_in > max_sum_S_j_in:
            max_sum_S_j_in = sum_S_j_in 
    lower = max_L_f + max_sum_S_j_in * max_L_g_s**2 + 1 + max_L_g**2
    
    return np.real(lower)

def lamb_lower(Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    m_sparse = Parameters['m_sparse']
    dim = Parameters['dim']
    lower = 0
    N = num_of_nodes * 1 + dim*num_of_nodes
    B = np.zeros((m_sparse*num_of_nodes,N))
    sparse_index = 0
    for i in range(num_of_nodes):
        if Data[i]['sparse_eq'] != []:
            num_neighbors = len(Data[i]['sparse_eq'])
            for j in range(num_neighbors):          
                B_i_s = Data[i]['A_j'][j]
                j_name = Data[i]['sparse_eq'][j]
                B[sparse_index*m_sparse:(sparse_index+1)*m_sparse,j_name*(dim+1):(j_name)*(dim+1)+dim] = B_i_s
            sparse_index +=1   
    return np.linalg.norm(B,ord=2)
    
################################################################################
                           #   Other Alg Part         
################################################################################

def theta_func(x_seq,Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    m = Parameters['m']

    theta = np.zeros((num_of_nodes,m))   
    for i in range(num_of_nodes):
        x_i = x_seq[i]

        aa_i = Data[i]['aa']
        cc_i = Data[i]['cc']
        A_i = Data[i]['A']

        theta[i,0] = np.linalg.norm(x_i-aa_i)**2 - cc_i

        if Data[i]['a_i'] != {}:
            S_j_in = Data[i]['a_i'].keys()
            for j in S_j_in:
                a_ji = Data[i]['a_i'][j]
                c_ji = Data[i]['c_i'][j]
                theta[i,1+j] += (np.linalg.norm(x_seq[i] - a_ji)**2 - c_ji)
        theta[i,1+num_of_nodes: 1+num_of_nodes+m_global] =  (A_i@x_seq[i])
        if Data[i]['A_i'] != {}:
            S_j_eq = Data[i]['A_i'].keys()
            for j in S_j_eq:
                A_ji = Data[i]['A_i'][j]
                theta[i,1+num_of_nodes+m_global+j*m_sparse:1+num_of_nodes+m_global+(j+1)*m_sparse] = (A_ji@x_seq[i])

    return theta

    
################################################################################
                           # Error Part         
################################################################################


def optimal_func_value(x,Data,part = 'smooth'):
    N = x.shape[0]

    function_value = 0 
    for i in range(N):
        P_i = Data[i]['P']
        Q_i = Data[i]['Q']

        function_value += (x[i].T@P_i@x[i] + Q_i.T@x[i])
        if part == 'l1':
            function_value += np.linalg.norm(x[i],1)
    return function_value



def violation_error(x,Data,Parameters):
    '''
    sum_constraint[0] <= 0
    sum_constraint[ 1 : num_of_nodes + 1] <=0

    sum_constraint[num_of_nodes + 1 : num_of_nodes + 1 + m_global] == 0
    sum_constraint[num_of_nodes + 1 + m_global : num_of_nodes + 1 + m_global + num_of_nodes * m_sparse] == 0

    '''
    
    num_of_nodes = Parameters['num_of_nodes']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']

    
    m_in = 1 + num_of_nodes
    
    for i in range(num_of_nodes):
        indicator_error = np.linalg.norm(x[i]-Data[i]['a'])**2 - Data[i]['c']
        if indicator_error <= 1e-6:
            indicator_error = 0
        else:
            print('Indicator function is not satisfied!!!')
            print(indicator_error)
        

    constraint = theta_func(x,Data,Parameters)
    #print(constraint)
    sum_constraint = np.sum(constraint,axis = 0)



    sum_constraint[:m_in] *= (sum_constraint[:m_in] >0)
    e_1 = sum_constraint[0]
    e_3 = np.sum(sum_constraint[1:num_of_nodes+1])
    e_2 = np.linalg.norm(sum_constraint[num_of_nodes+1:num_of_nodes + 1 + m_global])
    e_4 = 0
    for i in range(num_of_nodes):
        e_4 += np.linalg.norm(sum_constraint[num_of_nodes + 1 + m_global+i*m_sparse:num_of_nodes + 1 + m_global+(i+1)*m_sparse])


    return e_1+e_2+e_3+e_4

    


################################################################################
                           # IO Part         
################################################################################

def write_x_seq(file_name,sequence):

    # reshaping the array from 3D matrice to 2D matrice.
    arrReshaped = sequence.reshape(sequence.shape[0], -1)
    # saving reshaped array to file.
    np.savetxt(file_name, arrReshaped)
    return

def read_x_seq(file_name,dim):
    #to read file you saved
    loadedArr = np.loadtxt(file_name)
    # This loadedArr is a 2D array, therefore we need to convert it to the original array shape.
    # reshaping to get original matrice with original shape.
    loadedOriginal = loadedArr.reshape(loadedArr.shape[0], loadedArr.shape[1] // dim, dim)

    return loadedOriginal

def metro_generate(num_of_nodes,adjacency_matrix):
    metropolis=np.zeros((num_of_nodes,num_of_nodes))
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if adjacency_matrix[i,j]==1:
                d_i = np.sum(adjacency_matrix[i,:])
                d_j = np.sum(adjacency_matrix[j,:])
                metropolis[i,j]=1/(1+max(d_i,d_j))
        metropolis[i,i]=1-sum(metropolis[i,:])
    return metropolis