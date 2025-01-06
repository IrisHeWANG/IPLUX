import numpy as np
from utils import metropolis
import cvxpy as cp
from scipy.linalg import block_diag
from utils import optimal_func_value,violation_error
import copy
import time

def r_val(x_seq,Data):
    num_of_nodes,dim = x_seq.shape
    r_seq = np.zeros((num_of_nodes,dim))

    for i in range(num_of_nodes):
        A_i = Data[i]['A_i'] #record j:V^eq: i\in S_j^eq and the corresponding A_ji
        if A_i != {}: # i belongs to some S_j^eq
            j_set = A_i.keys() # the arrary constains:  j:V^eq: i\in S_j^eq
            for j in j_set:
                A_ji = A_i[j]
                S_j_eq = Data[j]['sparse_eq']
                A_j = Data[j]['A_j']
                temp = 0
                for l in range(len(S_j_eq)):
                    A_jl = A_j[l]
                    l_name = S_j_eq[l]
                    temp += (A_jl@x_seq[l_name])
                r_seq[i] += ((A_ji.T)@(temp))

    return r_seq

def s_val(x_seq,t_seq,Data):
    num_of_nodes,dim = x_seq.shape
    tilde_p = t_seq.shape[1]
    s_seq = []
    for i in range(num_of_nodes):
        if Data[i]['sparse_in'] == []:
            s_i = np.zeros(tilde_p)
            aa_i = Data[i]['aa']
            cc_i =  Data[i]['cc']
            x_i = x_seq[i]
            s_i[0] = np.linalg.norm(x_i - aa_i)**2 - cc_i - t_seq[i][0]
            
            s_seq.append(s_i)
        else:
            s_i = np.zeros(tilde_p + 1)
            aa_i = Data[i]['aa']
            cc_i =  Data[i]['cc']
            x_i = x_seq[i]

            a_j = Data[i]['a_j']
            c_j = Data[i]['c_j']
            S_i_in = Data[i]['sparse_in']

            s_i[0] = np.linalg.norm(x_i - aa_i)**2 - cc_i - t_seq[i][0]

            for j in range(len(S_i_in)):
                j_name = S_i_in[j]
                s_i[-1] += (np.linalg.norm(x_seq[j_name] - a_j[j])**2 - c_j[j])
            s_seq.append(s_i)

    return s_seq

def u_val(u_seq,x_seq,t_seq,z_seq,Data,rho,P_W):
    num_of_nodes,dim = x_seq.shape
    tilde_p = t_seq.shape[1]
    
    y_seq = np.concatenate((x_seq, t_seq), axis=1)
    u_new = np.zeros_like(u_seq)
    for i in range(num_of_nodes):
        A_i = Data[i]['A']
        B_i = block_diag(A_i, np.identity(tilde_p))
        u_new[i] = (B_i@y_seq[i]-z_seq[i])/rho

        for j in range(num_of_nodes):
            if P_W[i,j] != 0:
                u_new[i] += (P_W[i,j]*u_seq[j])
    return u_new

def z_val(z_seq,u_seq,rho,P_H):
    num_of_nodes,_ = z_seq.shape

    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if P_H[i,j]!= 0:
                z_seq[i] += (rho*P_H[i,j]*u_seq[j])
    return z_seq


def q_init(s_seq):
    num_of_nodes = len(s_seq)
    q_seq = []
    for i in range(num_of_nodes):
        q_i = np.maximum(-s_seq[i],np.zeros_like(s_seq[i]))
        q_seq.append(q_i)
    return q_seq

def q_val(q_seq,s_seq):
    num_of_nodes = len(s_seq)
    #print(num_of_nodes)
    q_new = []
    for i in range(num_of_nodes):
        q_i = np.maximum(-s_seq[i],q_seq[i]+s_seq[i])
        q_new.append(q_i)
    return q_new

def x_val(x_seq,v_x_seq,r_seq,u_seq,z_seq,q_seq,s_seq,Data,Parameters,Algorithm_Parameters,P_W,part = 'smooth'):
    '''
     grad_f_i = 2P_ix + Q_i
     h_i:
      - part == 'smooth':None (+constraint).
      - part == 'l1': l1_norm of x (+constraint).
    '''
    num_of_nodes,dim = x_seq.shape
   
    m_global = Parameters['m_global']
    gamma = Algorithm_Parameters['gamma']
    lamb = Algorithm_Parameters['lamb']
    alpha = Algorithm_Parameters['alpha']
    rho = Algorithm_Parameters['rho']

    x_new = np.zeros_like(x_seq)
    for i in range(num_of_nodes):
        P_i = Data[i]['P']
        Q_i = Data[i]['Q']
        A_i = Data[i]['A']
        a_i = Data[i]['a']
        c_i = Data[i]['c']
        aa_i = Data[i]['aa']
        cc_i =  Data[i]['cc']
        var_i = cp.Variable(dim)

        grad_f_i = 2*P_i@x_seq[i]+Q_i
        if part == 'smooth':
            prob_i = grad_f_i.T@(var_i-x_seq[i])
        if part == 'l1':
            prob_i = grad_f_i.T@(var_i-x_seq[i]) + cp.norm(var_i,1)
        prob_i += v_x_seq[i].T@var_i
        prob_i += (gamma*(lamb**2)/2*(cp.norm(var_i-x_seq[i]+r_seq[i]/(lamb**2))**2))
        prob_i += (cp.norm(A_i@var_i)**2/(2*rho))
        temp = 0
        for j in range(num_of_nodes):
            if P_W[i,j]!=0:
                temp += P_W[i,j]*u_seq[j][:m_global]
        prob_i += (temp - z_seq[i][:m_global]/rho).T@(A_i@var_i)
        prob_i += ((q_seq[i][0] + s_seq[i][0])* (cp.norm(var_i - aa_i)**2 - cc_i))

        if  Data[i]['a_i'] != {}:
            j_set = Data[i]['a_i'].keys()
            for j in j_set:
                a_ji = Data[i]['a_i'][j]
                c_ji = Data[i]['c_i'][j]
                prob_i += (((q_seq[j][-1] + s_seq[j][-1])*(cp.norm(var_i-a_ji)**2 - c_ji)))
        prob_i += (alpha/2 * (cp.norm(var_i - x_seq[i])**2))
        constraint = [cp.norm(var_i-a_i)**2 <= c_i]
        prob = cp.Problem(cp.Minimize(prob_i),constraint)
        prob.solve()

        x_new[i] = var_i.value
    return x_new
                
               

def t_val_1(t_seq,u_seq,z_seq,q_seq,s_seq,Data,Parameters,Algorithm_Parameters,P_W,adjacency_matrix):
    num_of_nodes = Parameters['num_of_nodes']
    m_global = Parameters['m_global']
    tilde_p = t_seq.shape[1]

    gamma = Algorithm_Parameters['gamma']
    lamb = Algorithm_Parameters['lamb']
    alpha = Algorithm_Parameters['alpha']
    rho = Algorithm_Parameters['rho']

    t_new = np.zeros_like(t_seq)
    for i in range(num_of_nodes):
        temp = 0 
        for j in range(num_of_nodes):
            if P_W[i,j] != 0:
                temp -= (P_W[i,j]*u_seq[j][m_global:])
        temp += ((gamma*(lamb**2)+alpha)*t_seq[i] + z_seq[i][m_global:]/rho + q_seq[i][:tilde_p] + s_seq[i][:tilde_p])
    
        t_new[i] = (1/((1/rho) + gamma* (lamb**2)+ alpha))* temp
    return t_new

def v_val(v_seq,r_seq,Parameters,gamma):
    num_of_nodes =  Parameters['num_of_nodes']
    v_new = np.zeros_like(v_seq)

    for i in range(num_of_nodes):
        v_new[i] = v_seq[i] + gamma * r_seq[i]
    return v_new

def IPLUX(max_iteration,Data,Parameters,Algorithm_Parameters,adjacency_matrix,opt_value,x_init,part = 'smooth'):
    print('\nRunning with IPLUX ... ')
    print("IPLUX:",Algorithm_Parameters) 
    '''
    Notation Correspondence
    num_of_nodes : d
    dim : d_i, for all i in V
    m_global : tilde{m}
    m_sparse : m_i, for all i in V
    Note that here tilde_p = 1 and p_i = 1.
        N = num_of_nodes * (1 + dim)
    '''
    #### Parameter Retrieval
    num_of_nodes = Parameters['num_of_nodes']
    dim = Parameters['dim']
    m_global = Parameters['m_global']
    m_sparse = Parameters['m_sparse']
    tilde_p = 1
    p_i = 1

    alpha = Algorithm_Parameters['alpha']
    lamb = Algorithm_Parameters['lamb']
    gamma = Algorithm_Parameters['gamma']
    rho = Algorithm_Parameters['rho']
   
    PP = metropolis(adjacency_matrix)
    np.savetxt('./final_data/PP', PP,fmt='%.2f')
    P_W = (np.identity(num_of_nodes) + PP)/2
    P_H = (np.identity(num_of_nodes) - PP)/2
    np.savetxt('./final_data/P_H', P_H,fmt='%.2f')
    np.savetxt('./final_data/P_W', P_W,fmt='%.2f')

    '''
    print('\n Check properties:')
    print('eigen of P_H',np.linalg.eigvals(P_H))
    print('eigen of P_W',np.linalg.eigvals(P_W))
    print('P_H+P_W = ',P_H+P_W)
    print('P_H@1 = ',np.sum(P_H,axis = 1))
    print('P_W@1 = ',np.sum(P_W,axis = 1))
    '''

  
    #### Initialization
    v_x_seq = np.zeros((num_of_nodes,dim))
    x_seq = copy.deepcopy(x_init)
    #x_seq = init_x(Data,Parameters)
    #x_seq = local_opt(Data,Parameters)
    t_seq = np.zeros((num_of_nodes, tilde_p))
    u_seq = np.zeros((num_of_nodes, m_global + tilde_p))
    z_seq = np.zeros((num_of_nodes, m_global + tilde_p))
    r_seq = r_val(x_seq, Data)
    s_seq = s_val(x_seq, t_seq, Data)
    q_seq = q_init(s_seq)

    average_x_seq = copy.deepcopy(x_seq)

    error_seq = [abs(optimal_func_value(x_seq,Data,'l1') - opt_value)]
    violation_error_seq = [violation_error(x_seq,Data,Parameters)]
    print('Iteration','0 :',error_seq[-1],violation_error_seq[-1])
    error_average = [abs(optimal_func_value(average_x_seq,Data,'l1') - opt_value)]
    violation_average = [violation_error(average_x_seq,Data,Parameters)]

    hist_x = np.zeros((max_iteration+1,num_of_nodes,dim))
    hist_x[0] = x_seq
    #### Update Flow
    k = 1
    init_time = time.time()
    while k<= max_iteration:
        if part == 'l1':
            x_seq = x_val(x_seq,v_x_seq,r_seq,u_seq,z_seq,q_seq,s_seq,Data,Parameters,Algorithm_Parameters,P_W,'l1')
        else:
            x_seq = x_val(x_seq,v_x_seq,r_seq,u_seq,z_seq,q_seq,s_seq,Data,Parameters,Algorithm_Parameters,P_W)
        t_seq = t_val_1(t_seq,u_seq,z_seq,q_seq,s_seq,Data,Parameters,Algorithm_Parameters,P_W,adjacency_matrix)
        r_seq = r_val(x_seq,Data)
        s_seq = s_val(x_seq,t_seq,Data)
        v_x_seq = v_val(v_x_seq,r_seq,Parameters,gamma)
        q_seq = q_val(q_seq,s_seq)
        u_seq = u_val(u_seq,x_seq,t_seq,z_seq,Data,rho,P_W)     
        z_seq = z_val(z_seq,u_seq,rho,P_H)


       
        average_x_seq = (k-1)/k*average_x_seq + x_seq/k
        hist_x[k] = x_seq
        if part == 'l1':
            error_seq.append(abs(optimal_func_value(x_seq,Data,'l1') - opt_value))
            error_average.append(abs(optimal_func_value(average_x_seq,Data,'l1') - opt_value))
        else:
            error_seq.append(abs(optimal_func_value(x_seq,Data) - opt_value))
        violation_error_seq.append( violation_error(x_seq,Data,Parameters))
        violation_average.append( violation_error(average_x_seq,Data,Parameters))
        if k%1 == 0:
            #print('check z_seq:', np.sum(z_seq,axis = 0))
            #print('check q_seq:', np.sum(q_seq,axis = 0))

            print('Iteration',k,':',error_seq[-1],violation_error_seq[-1])
            if k==100 or k==200 or k==500 or k==1000:
                print("Time",time.time()-init_time)
        #if k==2:
            #np.savetxt('./initial_x', x_seq)
    
        k += 1
    return error_seq,violation_error_seq,hist_x,error_average,violation_average,v_x_seq,u_seq,z_seq,q_seq