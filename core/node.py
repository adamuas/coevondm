# A Node class for the Neural Network



class node:
    """ A class for the neural network"""


    def __init__(self,node_id,map_id, node_type = -1):
        """constructor that sets the node id and node type"""
        self.node_id = node_id; #id of the node
        self.map_id = map_id; #id on the connection matrix and other 
        self.type = node_type; #type (i.e. input, hidden, output)

        #computational node parameters
        self.weightFn = -1; #weight function
        self.nodeFn =-1; #node function
        self.fnParam = []; #function
        self.fitness = -1; #fitness of the node during coevolution
        
# A sub class of the Node class for the Neural Network
class inactive_node(node):
    """ A class for the neural network"""


    def __init__(self, node_id, map_id, node_type=-1):
        """constructor that sets the node id and node type
        :type self: object
        """
        node.__init__(self, node_id, map_id)
        self.node_id = node_id;
        self.type = node_type;
        self.weightFn = -1;
        self.nodeFn = -1;
        self.fnParam = [];

        
        
        
    
    
    
    
