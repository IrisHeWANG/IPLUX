import numpy as np


def generate_objective(Data,num_of_nodes,feature_dim):
    '''
    Data: Least Square Problem \sum_{i} x_i^T P_i x_i + Q_i.T x_i

    P_i: shape = (feature_dim, feature_dim) = (5, 5)
    Q_i: shape = (feature_dim) = (3,)
    '''
    for i in range(num_of_nodes):
        Q,R = np.linalg.qr(np.random.rand(feature_dim,feature_dim))
        diag_elem = np.random.rand(feature_dim)
        diag_elem[-1] = 0 
        P_i=Q.T@np.diag(diag_elem)@Q
        Data[i]['P']=P_i
        print("Check Semipositivity:",min(np.linalg.eigvals(P_i)))
        Data[i]['Q']=np.random.rand(feature_dim)*3

    return Data

def generate_global_constraint(Data,num_of_nodes,m_global,feature_dim):
    '''
    Inequality : ||x_i -a_i||^2 <= c_i
                 \sum_{i} ||x_i -a_i'||^2 - c_i'\le 0 

    Equality:    \sum_{i} A_ix_i = 0
    
                - A: (m_global,feature_dim) = (3, 5)
    '''
    for i in range(num_of_nodes):
        Data[i]['a'] = np.random.rand(feature_dim)
        Data[i]['c'] = np.random.rand()+ Data[i]['a'].T@Data[i]['a']
        Data[i]['aa'] = np.random.rand(feature_dim)
        Data[i]['cc'] = np.random.rand() + Data[i]['aa'].T@Data[i]['aa']
        Data[i]['A'] = np.random.rand(m_global,feature_dim)
    return Data
def generate_sparse_constraint(Data,num_of_nodes,m_sparse,feature_dim,sparse_eq,sparse_in):
    '''
    Inequality:  \sum_j\in sparse_in[i] ||x_j - a_ij||^2 -c_ij <= 0, for all i in spare_in
    Equality: \sum_j\in sparse_eq[j] A_ij^s x_j = 0 , for all i in sparse_eq

    items in Data[i]:
        - a_j: matrix, shape = (num_of_sparse_neighbors,feature_dim)
            - if i in V^in: a_ij = a_j[index] s.t. ||x_j - a_ij||^2 -c_ij <= 0
            - else: zero vector.
        - a_i: a dict. 
            - keys: as whose a_j. For k in V^in: there exists ||x_i - a_ji ||^2 - c_ji <= 0 
            - values: the corresponding a_ji
            - if i not belongs any S_j^in, for j in V^in: a_i = {}
        - c_j, c_i similar to a_j, a_i.
        - A_j, A_i similar to a_j, A_i.
            - A_j: A_ijx_j
            - A_i: A_jix_i

    '''
    for i in range(num_of_nodes):
        Data[i]['a_i'] = {}
        Data[i]['c_i'] = {}
        Data[i]['A_i'] = {}
    for i in range(num_of_nodes):
        Data[i]['sparse_in'] = sparse_in[i]
        num_neighbors = len(Data[i]['sparse_in'])
        Data[i]['a_j'] = np.random.rand(num_neighbors,feature_dim)
        Data[i]['c_j'] = np.random.rand(num_neighbors)
        for j in range(num_neighbors):
            Data[i]['c_j'][j] += Data[i]['a_j'][j].T@Data[i]['a_j'][j] # To ascertain 0 in feasibile set.
            i_n = sparse_in[i][j]
            Data[i_n]['a_i'][i] = Data[i]['a_j'][j]
            Data[i_n]['c_i'][i] = Data[i]['c_j'][j]
        

        Data[i]['sparse_eq'] = sparse_eq[i]
        num_neighbors = len(Data[i]['sparse_eq'])
        Data[i]['A_j'] = np.random.rand(num_neighbors,m_sparse,feature_dim)
        for j in range(num_neighbors):
            i_n = sparse_eq[i][j]
            Data[i_n]['A_i'][i] = Data[i]['A_j'][j]
    return Data


def generate_data(Parameters,sparse_eq,sparse_in):
    '''
    problem:  
        minimize \sum_{i} x_i^T P_i x_i + Q_i.T x_i +(one norm)
        subject to 
                 ||x_i -a_i||^2 <= c_i (h_i as indicator funcion)
                 
                 \sum_{i} ||x_i -a_i'||^2 - c_i' <= 0 
                 \sum_{i} A_ix_i = 0
                 \sum_j\in sparse_in[i] ||x_j - a_ij||^2 -c_ij <= 0, for all i in spare_in
                 \sum_j\in sparse_eq[j] A_ij^s x_j = 0 , for all i in sparse_eq
    '''
    num_of_nodes = Parameters['num_of_nodes']
    feature_dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']


    Data=[{} for i in range(num_of_nodes)]
    Data = generate_objective(Data,num_of_nodes,feature_dim)
    Data = generate_global_constraint(Data,num_of_nodes,m_global,feature_dim)
    Data = generate_sparse_constraint(Data,num_of_nodes,m_sparse,feature_dim,sparse_eq,sparse_in)
    return Data
