import numpy as np
import network_generator
from problem_generator import generate_data
from opt import opt_solver
import copy
import pandas as pd
import xlwt
import time
from IPLUX import IPLUX
from otheralg import dual_subgradient,tracking_admm,pdc_admm
from utils import write_x_seq,read_x_seq,lamb_lower,alpha_lower,optimal_func_value,check_linear_independent,metro_generate,A_s

################################################################################
                           # Basic Setting         
################################################################################

##############   Fix Seed      ###########
np.random.seed(1)

part = 'l1'             # This is used for the nonsmooth problem.
#part = 'drawing'       # This is used for drawing the graph.

print('This simulation is for', part,'!\n')

############## Network setting ###########
num_of_nodes = 20
num_of_edges = 40


############# Problem Setting ############
dim=3
num_local_samples = 3
m_global = 5
m_sparse = 5

m =  1 + num_of_nodes + m_global + num_of_nodes * m_sparse #the number of constraints (in and eq)

### For simplicity, ``Parameters'' contains all the setting par.
Parameters = {'num_of_nodes':num_of_nodes, 'dim':dim,'num_local_samples':num_local_samples,'m_global':m_global,'m_sparse':m_sparse,'m':m}
print('Basic Setting:',Parameters)


################################################################################
                           # Network Part         
################################################################################
print('#################################### \n            Network Part              \n####################################')
    

G,adjacency_matrix,sparse_eq,sparse_in = network_generator.generate_graph(num_of_nodes,num_of_edges)
metropolis_matrix = metro_generate(num_of_nodes,adjacency_matrix)
print(metropolis_matrix)
np.savetxt('./final_data/graph_adj', adjacency_matrix,fmt='%d') # Store the adjacency matrix in file.
################################################################################
                           # Problem Part         
################################################################################
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
np.random.seed(1)
if part != 'drawing':
    print('#################################### \n            Problem Part              \n####################################')

    Data = generate_data(Parameters,sparse_eq,sparse_in)
    check_linear_independent(Data,Parameters)
A_s_all = A_s(Data,Parameters)
np.savetxt('./final_data/A_s', A_s_all,fmt='%.2f')
#################################################################################
                  # Optimal Solution Calculation Part
#################################################################################         
if part == 'l1':
    writer_problem = pd.ExcelWriter('./final_data/nonsmooth_problem_final.xls')
    df_problem=pd.DataFrame(Data)
    df_problem.to_excel(writer_problem,sheet_name='ProblemSetting')

    opt_solution,opt_value=opt_solver(Data,Parameters,sparse_in,sparse_eq,'l1')
    df_opt=pd.DataFrame({'nonsmooth_opt_solution':opt_solution,'nonsmooth_opt_value':opt_value})
    df_opt.to_excel(writer_problem,sheet_name='OptimalSolution')
    writer_problem.save()

#################################################################################
                  # Nonsmooth Simulation
#################################################################################   
print("###########Experiments Is Running NOW!")
if part == 'l1':
    max_iteration = 2000
    
    x_init = np.loadtxt('./initial_x')

    ####PDC-ADMM
    
    c=0.01 
    init_time = time.time()
    pdc_error,pdc_violation_error,pdc_hist = pdc_admm(max_iteration,Data,Parameters,opt_value,c,x_init)
    print("Running time for PDC-ADMM =",time.time()-init_time)
    write_x_seq('./final_data/pdc_hist_final',pdc_hist)
 

    ####Tracking -ADMM
    c=3
    init_time = time.time()
    tracking_error,tracking_violation_error,tracking_hist = tracking_admm(max_iteration,Data,Parameters,opt_value,metropolis_matrix,c,x_init)
    print("Running time for Tracking-ADMM =",time.time()-init_time)
    write_x_seq('./final_data/tracking_hist_final',tracking_hist)

    #### SG 
    gamma=50
    init_time = time.time()
    S_error,S_violation_error,S_hist = dual_subgradient(max_iteration,Data,Parameters,opt_value,adjacency_matrix,gamma,x_init)
    print("Running time for Subgradient =",time.time()-init_time)
    write_x_seq('./final_data/sub_hist_final',S_hist)
    
    #### IPLUX
    #x_init = np.loadtxt('./initial_x')
    
    
    IPLUX_Parameters = {'alpha':alpha_lower(Data,Parameters),'lamb':lamb_lower(Data,Parameters),'rho':1,'gamma':0.5}
    init_time = time.time()
    IPLUX_error,IPLUX_violation,IPLUX_hist,error_average,violation_average,v_x_seq,u_seq,z_seq,q_seq = IPLUX(max_iteration,Data,Parameters,IPLUX_Parameters,adjacency_matrix,opt_value,x_init,'l1')
    print("Running time for IPLUX =",time.time()-init_time)
    write_x_seq('./final_data/IPLUX_hist_final',IPLUX_hist)
    writer_error=pd.ExcelWriter("./final_data/IPLUX.xls")
    df_error= pd.DataFrame({'func_error':IPLUX_error,'violation_error':IPLUX_violation,'error_average':error_average,'violation_average':violation_average})
    df_error.to_excel(writer_error,sheet_name='IPLUX_error')
    writer_error.save() 
    np.savetxt('./final_data/IPLUX_v', v_x_seq)
    np.savetxt('./final_data/IPLUX_u', u_seq)
    np.savetxt('./final_data/IPLUX_z', z_seq)
    np.savetxt('./final_data/IPLUX_q', q_seq)




 
    
    #### Save error of all three algorithms to error.xls
    writer_error=pd.ExcelWriter("./final_data/nonsmooth_error_final.xls")
    df_error= pd.DataFrame({'IPLUX_error':IPLUX_error,'S_error':S_error,'Tracking_error':tracking_error,'PDC_error':pdc_error})
    df_error.to_excel(writer_error,sheet_name='FunctionValueError')
    
    df_feasibility= pd.DataFrame({'IPLUX_Error':IPLUX_violation,'S_violation_error':S_violation_error,'tracking_violation_error':tracking_violation_error,'PDC_violation_error':pdc_violation_error})
    df_feasibility.to_excel(writer_error,sheet_name='ViolationError')
    writer_error.save()   

#################################################################################
                  # Fine Tune
#################################################################################   
    ####PDC-ADMM
    '''
    writer_error=pd.ExcelWriter("./adjustpar/new_pdc/nonsmooth_pdc1.xls")
    c_list1 = [1e-2,5e-3,1e-3,5e-4]
    c_list2=[1,0.5,0.1,0.05]
    for c in c_list1:
        
        print("PDC-ADMM: c = ",c)
        pdc_error,pdc_violation,pdc_hist = pdc_admm(max_iteration,Data,Parameters,opt_value,c)
        df_error= pd.DataFrame({'PDC_FuncE':pdc_error,'PDC_VioE':pdc_violation})
        df_error.to_excel(writer_error,sheet_name="c="+str(c))
        writer_error.save()
    
    ####Tracking-ADMM
    #c=5e-1
    
    writer_error=pd.ExcelWriter("./adjustpar/new_tracking/nonsmooth_tracking1.xls")
    c_list1 = [10,8,3,1]
    c_list2 = [0.5,0.1,0.05,0.01]
    for c in c_list1:
        print("Tracking-ADMM: c = ",c)
        tracking_error,tracking_violation_error,tracking_hist = tracking_admm(max_iteration,Data,Parameters,opt_value,metropolis_matrix,c)
        df_error= pd.DataFrame({'PDC_FuncE':tracking_error,'PDC_VioE':tracking_violation_error})
        df_error.to_excel(writer_error,sheet_name="c="+str(c))
        writer_error.save()
    '''
    #### SG
    #gamma = 5
    '''
    writer_error=pd.ExcelWriter("./adjustpar/new_sub/nonsmooth_sub4.xls")
    gamma_list4 = [50,40]
    for gamma in gamma_list4:
        print("Subgradient: gamma = ",gamma)
        S_error,S_violation_error,S_hist = dual_subgradient(max_iteration,Data,Parameters,opt_value,adjacency_matrix,gamma)
        df_error= pd.DataFrame({'SG_FuncE':S_error,'SG_VioE':S_violation_error})
        df_error.to_excel(writer_error,sheet_name="gamma="+str(gamma))
        writer_error.save()

    
    #write_x_seq('./final_data/S_hist',S_hist)
    #### IPLUX
    #IPLUX_Parameters = {'alpha':alpha_lower(Data,Parameters)+4,'lamb':lamb_lower(Data,Parameters)+3,'rho':0.5,'gamma':0.1}
    #print("IPLUX:",IPLUX_Parameters)
    max_iteration=2000
    writer_error=pd.ExcelWriter("./adjustpar/new_iplux/nonsmooth_iplux21.xls")
    alpha_l = alpha_lower(Data,Parameters)
    lamb_l = lamb_lower(Data,Parameters)
    alpha_list = [alpha_l,alpha_l+2,alpha_l+4]
    lamb_list1 = [lamb_l]
    lamb_list2 = [lamb_l+2]
    rho_list = [0.1,0.5,1]
    gamma_list1 = [0.1]
    gamma_list2=[0.5,1]
    for alpha in alpha_list:
        for lamb in lamb_list2:
            for rho in rho_list:
                for gamma in gamma_list1:
                    IPLUX_Parameters = {'alpha':alpha,'lamb':lamb,'rho':rho,'gamma':gamma}
                    print("IPLUX:",IPLUX_Parameters)
                    IPLUX_error,IPLUX_violation,IPLUX_hist = IPLUX(max_iteration,Data,Parameters,IPLUX_Parameters,adjacency_matrix,opt_value,'l1')
                    df_error= pd.DataFrame({'IPLUX_FuncE':IPLUX_error,'IPLUX_VioE':IPLUX_violation})
                    name = "alpha"+str(int(alpha))+"lamb"+str(int(lamb))+"rho"+str(rho)+"gamma"+str(gamma)
                    df_error.to_excel(writer_error,sheet_name=name)
                    writer_error.save()    
    '''
    #write_x_seq('./final_data/IPLUX_hist',IPLUX_hist)
    #df_error= pd.DataFrame({'IPLUX_FuncE':IPLUX_error,'IPLUX_VioE':IPLUX_violation})
    #df_error.to_excel(writer_error)

    #writer_error.save()    
 
