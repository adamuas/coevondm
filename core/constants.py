#constants




NOT_EVALUATED = -1; #DO NOT CHANGE, (only change after modifying how genes are created in optimEnDec)
NONE = -1; #DO NOT CHANGE, (only change after modifying analysisKit.py)
YES = 1;
NO = 1;


#-node types
INPUT_NODE = 1;
HIDDEN_NODE = 2;
OUTPUT_NODE = 3;

#connection types
RANDOM = 1;
FULLY_CONNECTED = 2;

#-problem types
REGRESSION = 1;
CLASSIFICATION = 2;

#-node functions
IDENTITY = 1;
SIGMOID = 2;
GAUSSIAN = 3;
TANH = 4;
GAUSSIAN_II = 5;
PROB_SIGMOID = 6;
THIN_PLATE_SPLINE =7;
ARC_TAN = 8;


#-weight functions
INNER_PROD = 1;
EUCLID_DIST = 2;
HIGHER_ORDER_PROD = 3;
HIGHER_ORDER_SUB = 4;
STD_DEV = 5;
MIN = 6;
MAX = 7;
MANHATTAN_DIST = 8;
MAX_DIST = 9;
MEAN = 10;

#-status
ACTIVE = 1;
INACTIVE = 0;

#-count of meta-information genes
AGE_GENE = 3;
COST_GENE = 0;
COST2_GENE = 1;
MISC_GENE = 2;
META_INFO_COUNT = 4;

#-count of transfer function details (per node)
TRANS_INFO_COUNT = 5; #weightfn, nodefn, and 3 fnparameters


#Ensembles Selection methods
TOP_N = 1;
NON_DOMINATED_FRONT = 2;
GREEDY_SELECT = 3;



