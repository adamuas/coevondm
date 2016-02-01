import itertools;
import csvReader;
import numpy;
import csv;


#debug switch
debug = True;


def cross_validation_seperate(dataset,test_prop,write_to_file = False,name = None):
    """ seperates a given dataset into test set and training set according to a probability """
    
    #get the length of the dataset
    num_samples = len(dataset['IN']);
    print num_samples;
    point_of_split = round(test_prop * num_samples-1);
    
    #training set and test set
    train_set = dict();
    test_set = dict();
    
    #make training and test set
    train_set['IN'] = [];
    train_set['OUT'] = [];
    test_set['IN'] = [];
    test_set['OUT'] = [];

    
    #split
    #print dataset['IN'];
    #print dataset['OUT'];
    train_set['IN'] = dataset['IN'][int(point_of_split):];
    train_set['OUT'] = dataset['OUT'][int(point_of_split):];
    test_set['IN'] = dataset['IN'][:int(point_of_split)];
    test_set['OUT'] = dataset['OUT'][:int(point_of_split)];
    
    if debug == True:
        print "test_set:",len(test_set['IN']);
        print "train_set:",len(train_set['IN']);
        print "total:",len(dataset['IN']);
    
    
    #write training set to file
    #if write_to_file == True:
    #    if name == None:
    #        name = 'datasets.py'
    #    writer = csv.writer(open(name, 'wb'))
    #    for k,v in test_set.items():
    #        writer.writerow([k,v]);
        
        
    return train_set, test_set;


#read abalone dataset
#data = csvReader.read_csv_dataset('../datasets/abalone/abalone_norm.csv',',',1);
#data = csvReader.read_csv_dataset_right('../datasets/australian-credit-data/australian.csv',' ',14);
#train_set,test_set = cross_validation_seperate(data,0.25,True);
#print "ratio(test:train) = ", len(test_set['IN'])/float(len(train_set['IN']));


