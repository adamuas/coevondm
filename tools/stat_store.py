
#===============================================================================
# # This class is reponsible for saving statistics#
#===============================================================================



import pandas as pd;
import numpy as np;
import time;
import platform;

debug = False;
debug_section = True;




#===============================================================================
#  Stores statistics
#===============================================================================
def store_statistics( path = None, transfer_functions_likelihood = None, transfer_functions_assoc_err = None,
                        coexist_on_path_likelihood = None, coexist_in_layer_likelihood = None,
                        coexist_in_net_likelihood = None, conn_strength_likelihood = None,
                        conn_density_likelihood = None, weight_fns = None, node_fns = None ):
    #get pc name
    pc_name = platform.node();


    #create possible combinations
    #create transfer function combination
    combinations = [(w,n) for w in weight_fns for n in node_fns];

    if debug:
        print "combinations", combinations;
        
    print "connection density input (store_stat):",conn_density_likelihood;

    #store statistics
    data = dict();
    
    data['transfer_functions_likehood'] = pd.DataFrame(data = transfer_functions_likelihood, index = weight_fns, columns = node_fns);
    data['transfer_functions_assoc_err'] =  pd.DataFrame(data = transfer_functions_assoc_err, index = weight_fns, columns = node_fns);
    data['coexist_on_path_likelihood'] =  pd.DataFrame(data = coexist_on_path_likelihood, index = combinations, columns = combinations);
    data['coexist_in_layer_likelihood'] =  pd.DataFrame(data = coexist_in_layer_likelihood, index = combinations, columns = combinations);
    data['coexist_in_net_likelihood'] =  pd.DataFrame(data = coexist_in_net_likelihood, index = combinations, columns = combinations);
    data['conn_strength_likehood'] = pd.DataFrame(data = conn_strength_likelihood, index = combinations, columns = combinations);
    data['conn_density_likehood'] = pd.DataFrame(data = conn_density_likelihood, index = combinations);
 
    if debug :
        pd.options.display.max_columns = 50
        print "transfer_functions_likelihood:\n", data['transfer_functions_likehood'];
        print "transfer_functions_assoc_err:\n", data['transfer_functions_assoc_err'];
        print "coexist_on_path_likelihood:\n", data['coexist_on_path_likelihood'];
        print "coexist_in_layer_likelihood:\n", data['coexist_in_layer_likelihood'];
        print "coexist_in_net_likelihood:\n", data['coexist_in_net_likelihood'];
        print "conn_strength_likehood:\n", data['conn_strength_likehood'];
        print "conn_density_likehood:\n", data['conn_density_likehood'];
    
    #Store to csv
    timestamp = time.ctime();
    timestamp = timestamp.replace(' ','_');
    timestamp = timestamp.replace(':','');
    
    for k,v in data.iteritems():
        v.to_csv(path + '/'+  str(k)+ timestamp + '_'+pc_name +'.csv');
        
        
    
#===============================================================================
# #Test
#===============================================================================
#Random
#A = np.random.rand(7,5);
#B = np.random.rand(35,35)
#C = np.random.rand(1,35);
#
#store_statistics('../experiments',A,A,B,B,B,B,C, weight_fns = [1,2,3,4,5,6,7], node_fns = [1,2,3,4,5]);