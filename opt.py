import cvxpy as cp
import numpy as np


def opt_solver(Data,Parameters,sparse_in,sparse_eq,part = 'smooth'):
    '''
    for part == 'smooth' (default): calculate the opt value of smooth prob.
    for part == 'l1': calculate the opt value of nonsmooth prob.
        - ||x||_1 = \sum_{k=1}^{dim*num_of_nodes} |x_k| = \sum_{k=1}^{num_of_nodes} ||x_i||_1
    '''

    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    


    x = cp.Variable(dim*num_of_nodes)
    prob = 0
    sum_global_in = 0
    sum_global_eq = 0

    constraint=[]
    for i in range(num_of_nodes):
        P_i = Data[i]['P']
        Q_i = Data[i]['Q']
        a_i = Data[i]['a']
        c_i = Data[i]['c']
        aa_i = Data[i]['aa']
        cc_i = Data[i]['cc']
        A_i = Data[i]['A']
        a_ij = Data[i]['a_j']
        c_ij = Data[i]['c_j']
        A_ij = Data[i]['A_j']
        
        var_i=x[i*dim:(i+1)*dim]
        prob+=(cp.quad_form(var_i,P_i)+Q_i.T@var_i)

        #seperate: 
        constraint += [cp.norm(var_i-a_i)**2 <= c_i]
        
        #global inequality
        sum_global_in += (cp.norm(var_i - aa_i)**2 - cc_i)
        
        #global equality
        sum_global_eq += A_i@var_i

        #sparse inequality
        if i in sparse_in.keys():
            sum_sparse_in = 0
            num_neighbors = len(Data[i]['sparse_in'])
            for j in range(num_neighbors):
                i_n = sparse_in[i][j]
                var_j = x[i_n*dim:(i_n+1)*dim]
                sum_sparse_in += (cp.norm(var_j-a_ij[j])**2-c_ij[j])
            
            constraint += [sum_sparse_in <= 0]
        
        #sparse equality
        if i in sparse_eq.keys():
            sum_sparse_eq = 0
            num_neighbors = len(Data[i]['sparse_eq'])
            for j in range(num_neighbors):
                i_n = sparse_eq[i][j]             
                var_j = x[i_n*dim:(i_n+1)*dim]
                sum_sparse_eq += A_ij[j]@var_j
            constraint += [sum_sparse_eq == np.zeros(m_sparse)]
        
    constraint+=[sum_global_in <= 0, sum_global_eq == np.zeros(m_global)]
    
    if part == 'l1':
        prob += cp.norm(x,1)
    #print(prob,constraint)
    prob=cp.Problem(cp.Minimize(prob),constraint)
    prob.solve()

    print("\nThe optimal value = ", prob.value)
    print("A solution x = ", x.value)

    return x.value,prob.value