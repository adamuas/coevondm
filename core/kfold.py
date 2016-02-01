

import sys

sys.path.insert(0,"../");
#sys.path.insert(0,"optim");
sys.path.insert(0,"../datasets/");
sys.path.insert(0,"../../tools/");

import random;


""" for use with k-fold cross-validation"""
debug = False;
def kfold(D,numFolds,balanced = False,numClasses =2):
    """ divides a given dataset, D,  into k-folds, establishes the datasets to be used for each round"""
    
    #Dataset
    N = len(D['IN']);
    fold_size = round(N/numFolds);
    
    #store the datasets to be used for each round, access by kfoldround[i], where i is the round number
    kfoldRound = [];
    
    
    """(1) Shuffled the dataset """
    Dnew = shuffle(D);
    
    """(3)build dataset for each round using folds """
    rounds = numFolds;
    for i in xrange(rounds):
        
        #split
        if balanced:
            Folds = splitToBalancedFolds(Dnew,numFolds,numClasses);
        else:
            Folds = splitToFolds(Dnew,numFolds);
        
        if Folds == None:
            return None;

        #Get validation set
        dataset_valid = Folds.pop(i)
    
        if debug and False:
            print "*Validation Set:", dataset_valid;
            print "*Dataset Left:", Folds;
            
        #Get train set
        dataset_train = dict();
        dataset_train['IN'] = [];
        dataset_train['OUT'] = [];
        
        
        for f in Folds:
            dataset_train['IN'].extend(f['IN']);
            dataset_train['OUT'].extend(f['OUT']);
            
        #create tuple of dataset (train, valid) to be used for the round
        d = (dataset_train,dataset_valid);
        
        #append to datasets to be used for round i
        kfoldRound.append(d);
        
        if debug:
            print "#ROUND #",i;
            print "#Train Set:", dataset_train;
            print "#Test Set:", dataset_valid;
        
    return kfoldRound;

        
def splitToFolds(D,numFolds):
    """ splits the dataset into folds """
    #copy dataset
    Dnew = dict(D);
    #determine foldsize
    N = len(Dnew['IN']);
    foldSize = int((N/float(numFolds)));
    
    if foldSize <= 2:
        print "\n-PROBLEM: Ooops!!! Dataset is too small for cross-validation.";
        return None;
    
    if debug:
        print "N:", N;
        print "Fold Size:",foldSize;
        
    
    
    #store for folds
    folds = [];
    
    #start and end of folds
    start = 0;
    end = foldSize;
    
    for i in xrange(numFolds):
        """ split datasets """
        
        
        #create new fold
        fold = dict();
        try:
            fold['IN'] = Dnew['IN'][start:end];
            fold['OUT'] = Dnew['OUT'][start:end];
            
        except:
            # due to out of bound index, just pick all the rest as a fold
            fold['IN'] = Dnew['IN'][start:];
            fold['OUT'] = Dnew['OUT'][start:];
        
            
        
        #add fold
        if fold['IN'] != [] and fold['OUT'] != []:
            folds.append(fold);
        
        #increament fold position
        start += foldSize;
        end += foldSize;
        
    return folds;

def splitToBalancedFolds(D,numFolds,numClasses =2):
    """
    Splits a given dataset into the desired number of folds. However, unlike the other split to folds, the folds are
    supposed to be relatively more balanced.
    """

    #copy dataset
    Dnew = dict(D);
    #determine foldsize
    N = len(Dnew['IN']);
    foldSize = int((N/float(numFolds)));

    if foldSize <= 2:
        print "\n-PROBLEM: Ooops!!! Dataset is too small for cross-validation.";
        return None;

    if debug:
        print "N:", N;
        print "Fold Size:",foldSize;



    #store for folds
    folds = [];


    for i in xrange(numFolds):
        """ split datasets """


        #create new fold
        fold = dict();
        fold['IN'] = [];
        fold['OUT'] = [];
        try:

            while len(fold['OUT']) < foldSize:
                # Pick an example
                pass;




        except:
            raise Warning("Something went wrong while sampling datasets into balanced folds.");



        #add fold
        if fold['IN'] != [] and fold['OUT'] != []:
            folds.append(fold);

        #increament fold position
        start += foldSize;
        end += foldSize;

    return folds;


        
def shuffle(D):
    """ shuffles a given dataset ,D """
    
    Dnew = dict(D);
    
    LEN = len(Dnew['IN']);
    
    
    for pi,pattern in enumerate(Dnew['IN']):
            
        #pick random pattern
        ci = random.randint(0,LEN-1);
        while ci == pi:
            ci = random.randint(0,LEN-1);
            
        #switch places with random pattern
        Dnew['IN'][pi],Dnew['IN'][ci] = Dnew['IN'][ci],Dnew['IN'][pi];
        Dnew['OUT'][pi],Dnew['OUT'][ci] = Dnew['OUT'][ci],Dnew['OUT'][pi];
        
    return Dnew;




#TEST
# # trainset = optimParams.trainset;
# # #get the dataset
# # print ">>>BEFORE SHUFFLE SET:",trainset;
# # dataset = shuffle(trainset[0]);
# # print ">>>AFTER SHUFFLE SET",trainset;
#
# folds = splitToFolds(dataset,2);
# print ">>>FOLDS:";
# for i,f in enumerate(folds):
#    print "FOLD#",i,": ",f;
# print "FOLDS RAW:", folds;
# print folds.pop(0);
# kfold(trainset[0],3);