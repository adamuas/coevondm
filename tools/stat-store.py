#################################################
# This class is reponsible for saving statistics#
#################################################


import pandas;
import numpy;



def store_statistics(transfer_functions_likelihood = None, transfer_functions_assoc_err = None,
                        coexist_on_path_likelihood = None, coexist_in_layer_likelihood = None,
                        coexist_in_net_likelihood = None, conn_strength_likelihood = None,
                        conn_density_likelihood = None):

    #store statistics
    data = dict();
    
    data['transfer_functions_likehood'] = [];
    data['transfer_functions_assoc_err'] = [];
    data['coexist_on_path_likelihood'] = [];
    data['coexist_in_layer_likelihood'] = [];
    data['coexist_in_net_likelihood'] = [];
    data['conn_strength_likehood'] = [];
    data['conn_density_likehood'] = [];


          
          
          
        
          