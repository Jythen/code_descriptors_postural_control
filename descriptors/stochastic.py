
import statsmodels.api as sm
import numpy as np
from constants import labels


def SDA(signal, axis=labels.DIFF_ML):
    
    if not (axis in [labels.DIFF_ML, labels.DIFF_AP]):
        return {}
    
    time, msd = signal.get_signal(axis)

    frequency = signal.frequency


    log_time = np.log(time[1:])
    log_msd = np.log(msd[1:])

    ind_end_first_region = int(0.3*25) 
    list_rmse_s = []
    ind_start = int(0.3*25) + 1
    ind_stop = int(2.5*25)

    for i in range(ind_start, ind_stop+1):
        
        Y_s = log_msd[:i]
        X_s = sm.add_constant(log_time[:i])
        
        model_s = sm.OLS(Y_s,X_s)
        result_s = model_s.fit()


 
        rmse = np.sqrt( np.mean((result_s.resid)**2) )
        
        list_rmse_s.append(rmse)


    where_max = np.where(np.array(list_rmse_s)==np.min(np.array(list_rmse_s)))     
    ind_max = where_max[0][-1]   # list_rmse_s[ind_max] is the last highest R2    
    ind_end_first_region = ind_max + ind_start  


    list_rmse_l = []

    for i in range(ind_start, ind_stop+1):
        
        Y_s = log_msd[i-1:] 
        X_s = sm.add_constant(log_time[i-1:])


        model_s = sm.OLS(Y_s,X_s)
        result_s = model_s.fit()


        rmse = np.sqrt( np.mean((result_s.resid)**2) )
       
        list_rmse_l.append(rmse)

    where_max = np.where(np.array(list_rmse_l)==np.min(np.array(list_rmse_l)))     
    ind_max = where_max[0][-1]   # list_rmse_s[ind_max] is the last highest R2    
    ind_begin_second_region = ind_max + ind_start   



    Y_s = log_msd[:ind_end_first_region] 
    X_s = sm.add_constant(log_time[:ind_end_first_region])


    
    model_s = sm.OLS(Y_s,X_s)
    result_s = model_s.fit()

    
    params_log_s = result_s.params # only params are used



    Y_l = log_msd[ind_begin_second_region-1:] #Because log msd begins one point later
    X_l = sm.add_constant(log_time[ind_begin_second_region-1:])

    model_l = sm.OLS(Y_l,X_l)
    result_l = model_l.fit()

    
    params_log_l = result_l.params # only params are used
    

#
#
    log_critical_time = (params_log_l[0] - params_log_s[0]) / (params_log_s[1] - params_log_l[1])
    
    
    if log_critical_time > log_time[-1]:
        log_critical_time = log_time[-1]
    
    critical_time = np.exp(log_critical_time)



    log_critical_displacement =  params_log_s[0] + params_log_s[1] * log_critical_time
    critical_displacement = np.exp(log_critical_displacement)
    

   
    short_time_diffusion = np.exp(params_log_s[0]) 
    long_time_diffusion = np.exp(params_log_l[0])  
    short_time_scaling = params_log_s[1]/2
    long_time_scaling = params_log_l[1]/2
    

        
#   
    return {'short_time_diffusion'+'_'+axis : short_time_diffusion,
            'long_time_diffusion'+'_'+axis : long_time_diffusion,
            'critical_time'+'_'+axis : critical_time,
            'critical_displacement'+'_'+axis : critical_displacement,
            'short_time_scaling'+'_'+axis: short_time_scaling,
            'long_time_scaling'+'_'+axis : long_time_scaling}    

all_features = [SDA]
