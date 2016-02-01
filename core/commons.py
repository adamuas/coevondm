

import numpy;
import constants;
import time;

"""
Contains functions common to the rest of the files.
"""

debug = True;

def getStrTimeStamp():
    """returns the current time stamp as string"""
    timestamp = time.ctime();
    timestamp = timestamp.replace(' ','_');
    timestamp = timestamp.replace(':','');
    
    return timestamp;
    
def randomWithin(mn, mx):
    """returns a number with a range """
    rnd = mn + (numpy.random.rand() * mx);

    if(rnd > mx):
        rnd = mx;
    elif (rnd < mn):
        rnd = mn;

    return rnd;

def within(val ,mn, mx):
    """ """

    if val < mn :
         val = mn;
    if val > mx:
        val = mx;

    return val;
    
    
    
    
def matrixForm(lst, rows, cols):
    """ forms a list into a matrix  - returns a copy of the item"""

    mat = numpy.zeros((rows,cols));

    col_start = 0;
    col_end = cols;
    for row_i in xrange(rows):
        #copy number of elements for the row
        mat[row_i][:] = list(lst[col_start:col_end]);

        #update start and ends
        col_start += cols;
        col_end += cols;


    return mat;

            
def matrixToList(mat):
    """ forms a list from a matrix """
    
    lst = [];
    
    for row in mat:
        lst.extend(row);
        
    return lst;
                    
            
def lenInnerList(lst):
    """returns the length of inner list elements """

    count = 0;

    for x in lst:
        count += len(x);

    return count;

def realToBinary(lst):
    """ corrects of list from real values to binaries"""

    for li in xrange(len(lst)):
        lst[li] = abs(round(lst[li]));

    return lst;


def normaliseToRange(lst):
    """ puts values of a list within range """
    for li in xrange(len(lst)):
        if lst[li] < netParams.param_limits['pMin']:
            lst[li] = netParams.param_limits['pMin'];
            
        if lst[li] > netParams.param_limits['pMax']:
            lst[li] = netParams.param_limits['pMax'];
            
def flipBit(indx, lst = None):
    """ flips the bit at the index in the list"""
    if lst[indx] == 0:
        
        if lst== None:
            return 1;
        else:
            lst[indx] = 1;
    
    if lst[indx] == 1:
        if lst == None:
            return 0;
        else:
            lst[indx] == 0;

def putInRange(geneVal, minVal, maxVal):
    #puts value within range
    if geneVal < minVal:
        geneVal = minVal;
    if geneVal > maxVal:
        geneVal = maxVal;
        
    return geneVal;  
        

def getGenerator(Lst):
    """ creates a generator for a given list"""

    for item in Lst:
        yield item;
    

def countWithingRange(outs,lower_bound,upper_bound):
    """ couts the frequencies of values within the given range"""
    count = 0;
    
    for x in outs:
        if x >= lower_bound and x <= upper_bound:
            count +=1;
            
    return count;
    
""" METRICS """
""" DIVERSITY MEASURES """


def getErrs(sols):
    """ gets the error distribution of a given set of solutions """
    
    #errors
    errors = sols[:][constants.COST_GENE];
    
    print "errors",errors;
    
    return errors;