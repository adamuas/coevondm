
import networkx as nx;
import numpy as np;
import matplotlib.pylab as plt;
import matplotlib.cm as cm;
import pandas as pd;
import time;
import platform;

def path_analysis_1(CoexistMat, ConnStrengthMat, weightFns, nodeFns,num, show = False):
    """ Performs path analysis on a given coexistence and connection strength matrix"""


    #Check
    if weightFns == None:
        weightFns = netParams.weightFns;
    if nodeFns == None:
        nodeFns = netParams.nodeFns;
        
    
    #create transfer function combination
    Comb = [(w,n) for w in weightFns for n in nodeFns];
    CombLabels =[str(x) for x in Comb];
    
    nums = range(0,35);
    
    labels = dict(zip(nums,CombLabels));

    #create graph obj
    g = nx.DiGraph();

    #Settings
    #Settings
    node_size = 1500;
    node_alpha = 0.3;
    node_color =  '#CCFFFF';
    node_text_size = 12;
    edge_color = 'blue';
    edge_alpha = 0.3;
    edge_tickness=1;
    edge_text_pos = 0.3;
    text_font = 'sans-serif';
    
   
    #get size of the matrix
    if type(CoexistMat) == type(np.zeros(1)):
        I,J = CoexistMat.shape;
        #get mean value - for thresholding
        mean_coexist = CoexistMat.mean();
        
        
        if I != J :
            print "Error - Matrices must be square";
            return;
    else:
        #Pandas dataframe
        I,J = CoexistMat.shape;
        mean_coexist = CoexistMat.mean();
        mean_coexist = mean_coexist.mean();
        max_coexist = CoexistMat.max();
        max_coexist = max_coexist.max();
        
        t_coexist = mean_coexist + (max_coexist/4);
        
    if type(ConnStrengthMat) == type(np.zeros(1)):
        I2,J2 = ConnStrengthMat.shape;
        mean_connstrength = ConnStrengthMat.mean();
        
        if I != J :
            print "Error - Matrices must be square";
            return;
    else:
        #pandas dataframe
        I2,J2 = ConnStrengthMat.shape;
        mean_connstrength = ConnStrengthMat.mean(); 
        mean_connstrength = mean_connstrength.mean();
        max_connstrength = ConnStrengthMat.max();
        max_connstrength = max_connstrength.max();
        
        t_connstrength = mean_connstrength + (max_connstrength/4);
    
    
    for node_i in xrange(I):
        for node_j in xrange(I):

            if type(ConnStrengthMat) == type(np.zeros(1)): #numpy
               
                #establish nodes
                g.add_node(node_i);
                g.add_node(node_j);
                if ConnStrengthMat[node_i,node_j] >= t_connstrength:
                    #establish edges
                    g.add_weighted_edges_from(
                        [(node_i, node_j ,CoexistMat[node_i,node_j]) ]
                         )
            else: #pandas dataframe
                
                #establish nodes
                g.add_node(node_i);
                g.add_node(node_j);
                if ConnStrengthMat.iloc[node_j,node_i] >= t_connstrength:
                    #establish edges
                    g.add_weighted_edges_from(
                        [(node_i, node_j ,CoexistMat.iloc[node_j, node_i]) ]
                         )
                    
    

    #plot
    colors = xrange(len(g.edges()));
    graph_pos=nx.shell_layout(g)
    nx.draw_networkx_nodes(g,graph_pos, node_size=node_size, alpha=0.7, node_color = node_color );
    nx.draw_networkx_edges(g,graph_pos, edge_color = colors, width = 2, edge_cmap = plt.cm.Blues, alpha=edge_alpha, arrows=True);
    nx.draw_networkx_labels(g,graph_pos, labels = labels, font_size = node_text_size,font_family = text_font , font_color  = '#003366' );
    plt.xticks([]);
    plt.yticks([]);
    plt.draw();
    
    # Get transfer function statistics
    timestamp = time.ctime();
    timestamp = timestamp.replace(' ','_');
    timestamp = timestamp.replace(':','');
    pc_name = platform.node();
    plt.savefig(str(num) + "_path_analysis"+timestamp+"_"+pc_name+ ".png",dpi=1000);
    
    if show:
        plt.show();
        
    plt.clf();
    plt.close();
    
    
    
def path_analysis_2(CoexistMat, ConnStrengthMat, weightFns, nodeFns, plot = True):
    """ Performs path analysis on a given coexistence and connection strength matrix"""


    #Check
    if weightFns == None:
        weightFns = netParams.weightFns;
    if nodeFns == None:
        nodeFns = netParams.nodeFns;
        
    
    #create transfer function combination
    Comb = [(w,n) for w in weightFns for n in nodeFns];
    CombLabels =[str(x) for x in Comb];
    
    nums = xrange(0,35);
    
    labels = dict(zip(nums,CombLabels));
   

    
    #create graph obj
    g = nx.DiGraph();

    #Settings
    node_size = 1600;
    node_alpha = 0.3;
    node_color =  '#CCFFFF';
    node_text_size = 12;
    edge_color = 'blue';
    edge_alpha = 0.3;
    edge_tickness=1;
    edge_text_pos = 0.3;
    text_font = 'sans-serif';
    
   
    #get size of the matrix
    if type(CoexistMat) == type(np.zeros(1)):
        I,J = CoexistMat.shape;
        #get mean value - for thresholding
        mean_coexist = CoexistMat.mean();
        
        
        if I != J :
            print "Error - Matrices must be square";
            return;
    else:
        #Pandas dataframe
        I,J = CoexistMat.shape;
        mean_coexist = CoexistMat.mean();
        mean_coexist = mean_coexist.mean();
        max_coexist = CoexistMat.max();
        max_coexist = max_coexist.max();
        
        t_coexist = mean_coexist + (max_coexist/4);
        
    if type(ConnStrengthMat) == type(np.zeros(1)):
        I2,J2 = ConnStrengthMat.shape;
        mean_connstrength = ConnStrengthMat.mean();
        
        if I != J :
            print "Error - Matrices must be square";
            return;
    else:
        #pandas dataframe
        I2,J2 = ConnStrengthMat.shape;
        mean_connstrength = ConnStrengthMat.mean(); 
        mean_connstrength = mean_connstrength.mean();
        max_connstrength = ConnStrengthMat.max();
        max_connstrength = max_connstrength.max();
        
        t_connstrength = mean_connstrength #+ (max_connstrength/4);
    
    idx_largest = -1;
    col_largest = -1;
    col_sum_max = -1;
    col_connn_ind = -1;
    #find the node with the largest connections
    for i,column in enumerate(CoexistMat.columns):
        
        #val = CoexistMat[column].idxmax();
        #if val > idx_largest:
        #    idx_largest = val;
        #    col_largest = column;
        #    col_connn_ind = column;
            
        col_sum = CoexistMat[column].sum();
        if col_sum > col_sum_max:
            col_sum_max = col_sum;
            col_connn_ind = column;
        
 
    #add the most connected terminal node
    h_conn_node = int(CoexistMat.columns.get_loc(col_connn_ind));
    g.add_node(str(col_connn_ind));
    for node_i,row in enumerate(CoexistMat[col_connn_ind]):
        
        
       if ConnStrengthMat.iloc[h_conn_node,node_i] >= t_connstrength:
        ##establish nodes
            g.add_node(labels[node_i]);
        ##establish edges
            g.add_weighted_edges_from(
                [(str(col_connn_ind), labels[node_i] ,ConnStrengthMat.iloc[h_conn_node, node_i]) ]
                 );
    ##plot
    colors = xrange(len(g.edges()));
    graph_pos=nx.shell_layout(g);
    nx.draw_networkx_nodes(g,graph_pos, node_size=node_size, alpha=0.7, node_color = node_color );
    nx.draw_networkx_edges(g,graph_pos, edge_color = colors, width = 2, edge_cmap = plt.cm.Blues, alpha=edge_alpha, arrows=True);
    nx.draw_networkx_labels(g,graph_pos, font_size = node_text_size,font_family = text_font , font_color  = '#003366' );
    plt.xticks([]);
    plt.yticks([]);
    plt.draw();
    #
    # Get transfer function statistics
    timestamp = time.ctime();
    timestamp = timestamp.replace(' ','_');
    timestamp = timestamp.replace(':','');
    pc_name = platform.node();
    plt.savefig("path_analysis"+timestamp+"_"+pc_name+ ".png",dpi=1000);
    
    if plot:
        plt.show();
    
    plt.clf();
    plt.close();
    
    
    
def test():
    weightFns = [1,2,3,4,5,6,7];
    nodeFns = [1,2,3,4,5];
    
    #create transfer function combination
    Comb = [(w,n) for w in weightFns for n in nodeFns];
    CombLabels =[str(x) for x in Comb];
    
    A = np.random.rand(35,35);
    B = np.random.rand(35,35);

    #path_analysis_1(A,B,[1,2,3,4,5,6,7], [1,2,3,4,5]);

    C = pd.DataFrame(A);
    D = pd.DataFrame(B);
    
    path_analysis_1(C,D, weightFns, nodeFns);



