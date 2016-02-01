import numpy;
import stats;


def areSimilarLst(A,B,level=0.05):
    """ finds the similarity of the given list and the next. default level is 0.05
    
        Returns True if they are similar, or returns False otherwise
    """
    
    #only for same length list
    if (len(A) != len(B)):
        return None;
    
    #set some parameters
    degree_of_freedom = len(A);
    significance_level = level; 
    
    
    #calculate chisquared between the two lists
    Exp_vals = A;
    Obv_vals = B;
    
    chiSquared = 0.0;
    
    #use the pearson chi squared test
    chiSquared,p_val = stats.lchisquare(Obv_vals,Exp_vals);
    
    print "chiSquared, p_val: ", chiSquared, p_val;
    if p_val <= significance_level:
        return True,chiSquared,p_val;
    else:
        return False,chiSquared,p_val;
       
    
#    
#W1 = numpy.random.rand(50).astype(numpy.float32);
#W2 = numpy.random.rand(50).astype(numpy.float32);
#W3 = numpy.random.poisson(1.0,50);
#print areSimilarLst(W1.tolist(),W2.tolist(),0.05);
#print areSimilarLst(W1.tolist(),W3.tolist(),0.05);