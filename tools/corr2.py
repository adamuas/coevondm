import numpy as np;



def corr2(A,B):
    """  """

    #get average of both matrices
    
    EA = np.mean(A);
    EB = np.mean(B);

    
    #have to be off same size
    if ((A.shape) != (B.shape)):
        print """- Cannot computer pearson correlation for the matrices:
                Have to be of same size""";
        return None;

 
    W, H = (A.shape);
    numerator = 0.0;
    devA = 0.0;
    devB = 0.0;
    for i in xrange(W):
        for j in xrange(H):

              Aij = A[i][j];
              Bij = B[i][j];

              numerator += (Aij - EA )* (Bij - EB);
              devA += (Aij - EA) * (Aij - EA);
              devB += (Bij - EB) * (Bij - EB);
              

    pearson_r = numerator/np.sqrt(devA * devB);

    

    return pearson_r;
    

##
##
### TEST
##A = np.random.rand(5,5) ;
##B = np.random.rand(5,5);
##print A;
##print B;
##corr2(A,A);
