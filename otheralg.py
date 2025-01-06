import numpy as np
from numpy.core.fromnumeric import var
import cvxpy as cp
import math
import copy
from utils import optimal_func_value,violation_error,theta_func
from scipy.linalg import block_diag
import time


#################################################################################
                  # PDC-ADMM
#################################################################################
def pdc_admm(max_iteration,Data,Parameters,opt_value,c,x_init):
    print('\nRunning with the PDC-ADMM Algorithm...')
    print("PDC-ADMM: c=",c)

    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    
    n = dim+1+num_of_nodes
    p = m_global+num_of_nodes*m_sparse + 1 + num_of_nodes
    ############# Tracking-ADMM Initialization
    x_seq = np.zeros((num_of_nodes,n))
    x_seq[:,:dim] = copy.deepcopy(x_init)
    x_new = np.zeros((num_of_nodes,n))
    y_seq = np.zeros((num_of_nodes,p))
    p_seq = np.zeros((num_of_nodes,p)) 
    y_new = np.zeros((num_of_nodes,p))
    p_new = np.zeros((num_of_nodes,p))

    ############# Initialization Error
    func_value_error_seq = [abs(optimal_func_value(x_seq[:,:dim],Data,'l1')-opt_value)]
    violation_error_seq=[violation_error(x_seq[:,:dim],Data,Parameters)]
    
    print('Iteration',0,':',func_value_error_seq[-1],violation_error_seq[-1])
    hist_x = np.zeros((max_iteration+1,num_of_nodes,dim))
    hist_x[0] = x_seq[:,:dim]
    E_new = A_new_gn(Data,Parameters)
    k = 1
    init_time = time.time()
    while k <= max_iteration:
        for i in range(num_of_nodes):
            x_new[i] = pdc_x(y_seq,Data,p_seq[i],i,c,dim,n,E_new[i])
            num_neighbor = len(Data[i]['sparse_eq'])-1
            temp = 1/c*(-p_seq[i]+E_new[i]@x_new[i])
            for j in Data[i]['sparse_eq'][1:]:
                temp += (y_seq[i] + y_seq[j])
            y_new[i]=1/(2*num_neighbor)*(temp)
        x_seq = copy.deepcopy(x_new)
        y_seq = copy.deepcopy(y_new)
        sum_p_i=0
        for i in range(num_of_nodes):
            temp =0
            for j in Data[i]['sparse_eq'][1:]:
                temp += (y_seq[i] - y_seq[j])
            p_seq[i] = p_seq[i] + c*temp
            sum_p_i += p_seq[i]
        #print('norm of sum_p,',np.linalg.norm(sum_p_i))
        func_value_error_seq.append(abs(optimal_func_value(x_seq[:,:dim],Data,'l1')-opt_value))
        violation_error_seq.append(violation_error(x_seq[:,:dim],Data,Parameters))

        hist_x[k] = x_seq[:,:dim]

        if k%1 == 0:

            print('Iteration',k,':',func_value_error_seq[-1],violation_error_seq[-1])
            if k==100 or k==200 or k==500 or k==1000:
                print("Time",time.time()-init_time)
        k+=1
        
    return func_value_error_seq,violation_error_seq,hist_x 

def pdc_x(y_seq,Data,p_i,i,c,dim,m,A_new_i):
    a_i = Data[i]['a']
    c_i = Data[i]['c']
    aa_i = Data[i]['aa']
    cc_i = Data[i]['cc']
    P_i = Data[i]['P']
    Q_i = Data[i]['Q']

    var_i = cp.Variable(m)
    x_i = var_i[:dim]
    num_neighbor = len(Data[i]['sparse_eq'])-1 
    #print(i,num_neighbor)
    prob_i = cp.quad_form(x_i,P_i)+Q_i.T@x_i + cp.norm(x_i,1)

    temp = 1/c*(A_new_i@var_i) - 1/c*p_i
    for j in Data[i]['sparse_eq'][1:]:
        temp += (y_seq[i] + y_seq[j])
    prob_i += c/(4*num_neighbor)*cp.norm(temp)**2

    constraint_i = [cp.norm(x_i-a_i)**2 <= c_i]
    constraint_i += [cp.norm(x_i - aa_i)**2 - cc_i<=var_i[dim]]
    for j in Data[i]['sparse_eq']:
        constraint_i += [cp.norm(x_i - Data[i]['a_i'][j])**2 - Data[i]['c_i'][j]<=var_i[dim+1+j]]
    
        
    prob = cp.Problem(cp.Minimize(prob_i),constraint_i)
    prob.solve(solver=cp.CPLEX)

    return var_i.value

#################################################################################
                  # Tracking-ADMM
#################################################################################
def tracking_admm(max_iteration,Data,Parameters,opt_value,laplacian_matrix,c,x_init):
    print('\nRunning with the Tracking-ADMM Algorithm...')
    print("Tracking-ADMM c=",c)
    W = laplacian_matrix
    A_new = A_new_gn(Data,Parameters)

    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    
    n = dim+1+num_of_nodes
    p = m_global+num_of_nodes*m_sparse + 1 + num_of_nodes
    ############# Tracking-ADMM Initialization
    x_seq = np.zeros((num_of_nodes,n))
    x_seq[:,:dim] = copy.deepcopy(x_init)
    x_new = np.zeros((num_of_nodes,n))
    lambda_seq = np.zeros((num_of_nodes,p))
    d_seq = np.zeros((num_of_nodes,p)) #when x=0
    lambda_new = np.zeros((num_of_nodes,p))
    d_new = np.zeros((num_of_nodes,p))
    for i in range(num_of_nodes):
        d_seq[i] = A_new[i]@x_seq[i]

    ############# Initialization Error
    func_value_error_seq = [abs(optimal_func_value(x_seq[:,:dim],Data,'l1')-opt_value)]
    violation_error_seq=[violation_error(x_seq[:,:dim],Data,Parameters)]
    
    print('Iteration',0,':',func_value_error_seq[-1],violation_error_seq[-1])
    hist_x = np.zeros((max_iteration+1,num_of_nodes,dim))
    hist_x[0] = x_seq[:,:dim]
    
    ############# Tracking-ADMM Updateflow
    k = 1
    init_time = time.time()
    while k <= max_iteration:
        for i in range(num_of_nodes):
            delta_i = 0
            l_i = 0
            for j in Data[i]['sparse_in']:
                #print("Wij:",W[i][j])
                delta_i += W[i][j]*d_seq[j]
                l_i += W[i][j]*lambda_seq[j]

            x_new[i] = tracking_x(x_seq[i],Data,l_i,delta_i,i,c,dim,n,A_new[i])
            #print(x_new[i])
            d_new[i] = delta_i + A_new[i]@x_new[i]-A_new[i]@x_seq[i]
            lambda_new[i] = l_i +c*d_new[i]
        x_seq = copy.deepcopy(x_new)
        d_seq = copy.deepcopy(d_new)
        lambda_seq = copy.deepcopy(lambda_new)
        func_value_error_seq.append(abs(optimal_func_value(x_seq[:,:dim],Data,'l1')-opt_value))
        violation_error_seq.append(violation_error(x_seq[:,:dim],Data,Parameters))

        hist_x[k] = x_seq[:,:dim]

        if k%1 == 0:

            print('Iteration',k,':',func_value_error_seq[-1],violation_error_seq[-1])
            if k==100 or k==200 or k==500 or k==1000:
                print("Time",time.time()-init_time)
        k+=1
        
        #print(k,func_value_error_seq[-1],violation_error_seq[-1],(optimal_func_value(x_seq,Data)-q)/num_of_nodes+gamma*k/(num_of_nodes**2*math.sqrt(k+1))*violation_error_seq[-1]**2/2)

    return func_value_error_seq,violation_error_seq,hist_x

def A_new_gn(Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    
    n = dim+1+num_of_nodes
    p = m_global+num_of_nodes*m_sparse + 1 + num_of_nodes
    A_new = np.zeros((num_of_nodes,p,n))
    for i in range(num_of_nodes):
        A_new[i][:m_global,:dim] = Data[i]['A']
        for j in Data[i]['sparse_in']:
            A_new[i][m_global+j*m_sparse:m_global+(j+1)*m_sparse,:dim] = Data[i]['A_i'][j]
            A_new[i][m_global+num_of_nodes*m_sparse+1+j] = 1
        A_new[i][m_global+num_of_nodes*m_sparse] = 1

    return A_new
            
        


def tracking_x(x_seq_i,Data,l_i,delta_i,i,c,dim,m,A_new_i):
    a_i = Data[i]['a']
    c_i = Data[i]['c']
    aa_i = Data[i]['aa']
    cc_i = Data[i]['cc']
    P_i = Data[i]['P']
    Q_i = Data[i]['Q']

    var_i = cp.Variable(m)
    x_i = var_i[:dim]
    #print(A_new_i)
    prob_i = cp.quad_form(x_i,P_i)+Q_i.T@x_i + cp.norm(x_i,1) + l_i.T@A_new_i@var_i+c/2*cp.norm(A_new_i@var_i-A_new_i@x_seq_i+delta_i)**2
    constraint_i = [cp.norm(x_i-a_i)**2 <= c_i]
    constraint_i += [cp.norm(x_i - aa_i)**2 - cc_i<=var_i[dim]]
    for j in Data[i]['sparse_eq']:
        constraint_i += [cp.norm(x_i - Data[i]['a_i'][j])**2 - Data[i]['c_i'][j]<=var_i[dim+1+j]]

    prob = cp.Problem(cp.Minimize(prob_i),constraint_i)
    prob.solve(solver=cp.CPLEX)

    return var_i.value
    


#################################################################################
                  # Dual Subgradient Simulation
#################################################################################
def balanced_weightmatrix(adjacency_matrix):
    N = adjacency_matrix.shape[0]

    P = (np.identity(N) + adjacency_matrix)
    degree = np.sum(adjacency_matrix,axis=1)
    for i in range(N):
        P[i,:] /= (degree[i] + 1)
        
    return P


def subgradient_x(lambda_seq,Data,Parameters):
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']

    x_new = np.zeros((num_of_nodes,dim))
    q = 0
    for i in range(num_of_nodes):  
        a_i = Data[i]['a']
        c_i = Data[i]['c']
        aa_i = Data[i]['aa']
        cc_i = Data[i]['cc']
        A_i = Data[i]['A']
        P_i = Data[i]['P']
        Q_i = Data[i]['Q']

        lambda_i = lambda_seq[i]
        prob_i = 0
        var_i = cp.Variable(dim)

        prob_i += lambda_i[0]*(cp.norm(var_i-aa_i)**2 - cc_i)

        if Data[i]['a_i'] != {}:
            S_j_in = Data[i]['a_i'].keys()
            for j in S_j_in:
                a_ji = Data[i]['a_i'][j]
                c_ji = Data[i]['c_i'][j]
                prob_i += lambda_i[1+j]*(cp.norm(var_i - a_ji)**2 - c_ji)
        prob_i += lambda_i[1+num_of_nodes: 1+num_of_nodes+m_global].T@ (A_i@var_i)
        if Data[i]['A_i'] != {}:
            S_j_eq = Data[i]['A_i'].keys()
            for j in S_j_eq:
                A_ji = Data[i]['A_i'][j]
                prob_i += lambda_i[ 1+num_of_nodes+m_global+j*m_sparse:1+num_of_nodes+m_global+(j+1)*m_sparse].T@(A_ji@var_i)
        constraint = [cp.norm(var_i-a_i)**2 <= c_i]
        prob = cp.Problem(cp.Minimize(cp.quad_form(var_i,P_i)+Q_i.T@var_i+cp.norm(var_i,1)+prob_i),constraint)
        prob.solve()

        x_new[i] = var_i.value
        q += prob.value

        
    return x_new,q
        

def dual_subgradient(max_iteration,Data,Parameters,opt_value,adjacency_matrix,gamma,x_init):
    print('\nRunning with the Dual_Subgradient Algorithm...')
    print("Subgradient: gamma=",gamma)
    P = balanced_weightmatrix(adjacency_matrix)

    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m = Parameters['m']
    
    m_in =  1 + num_of_nodes
    ############# Subgradient Initialization
    x_seq = copy.deepcopy(x_init)
    lambda_seq = np.zeros((num_of_nodes,m))
    z_seq = np.zeros((num_of_nodes,m))

    ############# Initialization Error
    func_value_error_seq = [abs(optimal_func_value(x_seq,Data,'l1')-opt_value)]
    violation_error_seq=[violation_error(x_seq,Data,Parameters)]
    
    print('Iteration',0,':',func_value_error_seq[-1],violation_error_seq[-1])
    hist_x = np.zeros((max_iteration+1,num_of_nodes,dim))
    hist_x[0] = x_seq
    ############# Subgradient Updateflow
    k = 1
    init_time = time.time()
    while k <= max_iteration:
        alpha = gamma/math.sqrt(k+1)

        x_hat,q=subgradient_x(lambda_seq,Data,Parameters)

        x_new= (k-1)/k*x_seq+x_hat/k

        theta_new = theta_func(x_new,Data,Parameters)
        theta_last = theta_func(x_seq,Data,Parameters)
        z_seq = P@z_seq + k*theta_new - (k-1)*theta_last

        temp = alpha*z_seq
        temp[:,:m_in] *= (temp[:,:m_in]>0)
        lambda_seq = (k-1)/(k)*lambda_seq + temp/(k)

        x_seq = copy.deepcopy(x_new)
        func_value_error_seq.append(abs(optimal_func_value(x_seq,Data,'l1')-opt_value))
        violation_error_seq.append(violation_error(x_seq,Data,Parameters))

        hist_x[k] = x_seq

        if k%1== 0:

            print('Iteration',k,':',func_value_error_seq[-1],violation_error_seq[-1])
            if k==100 or k==200 or k==500 or k==1000:
                print("Time",time.time()-init_time)
        k+=1
        
        #print(k,func_value_error_seq[-1],violation_error_seq[-1],(optimal_func_value(x_seq,Data)-q)/num_of_nodes+gamma*k/(num_of_nodes**2*math.sqrt(k+1))*violation_error_seq[-1]**2/2)

    return func_value_error_seq,violation_error_seq,hist_x

