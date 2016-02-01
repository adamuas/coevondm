import csv;
import ntpath;

debug = False;
verbose = False;

def read_csv_dataset(filename,delim,output_col):
    """reads dataset from csv file"""
    dataset = dict();
    
    #get the name of the file
    head,tail = ntpath.split(filename)
    
    #inputs
    dataset['NAME'] = tail;
    dataset['IN'] = [];
    dataset['OUT'] = [];
    
    with open(filename,'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=delim,quoting=csv.QUOTE_NONE);
        
        #data
        for x in data:
                inl = x[output_col:];
                outl = x[:output_col];
                dataset['IN'].append(inl);
                dataset['OUT'].append(outl);
            
        
    for i,v in enumerate(dataset['IN']):
        dataset['IN'][i] = [float(x) for x in v];
        #print dataset['IN'][i];
    for i,v in enumerate(dataset['OUT']):
        dataset['OUT'][i] = [float(x) for x in v];
        #print dataset['OUT'][i];
        

    return dataset;

def read_csv_dataset_right(filename,delim,output_col):
    """reads dataset from csv file"""
    dataset = dict();
    
    #get the name of the file
    head,tail = ntpath.split(filename)
    
    #inputs
    dataset['NAME'] = tail;
    dataset['IN'] = [];
    dataset['OUT'] = [];
    
    with open(filename,'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=delim,quoting=csv.QUOTE_NONE);
        
        #data
        for x in data:
                outl = x[output_col:];
                inl = x[:output_col];

                if inl != []:

                    dataset['IN'].append(inl);
                    dataset['OUT'].append(outl);
            
        
    for i,v in enumerate(dataset['IN']):
        
        if debug:
            print i, v;
            print [float(x) for x in v];
            
        #conver to float
        dataset['IN'][i] = [float(x) for x in v];
            

    for i,v in enumerate(dataset['OUT']):
        dataset['OUT'][i] = [float(x) for x in v];
        
    
    if verbose or debug:
        print dataset;

    return dataset;


def read_csv_dataset_left(filename,delim):
    """reads dataset from csv file"""
    dataset = dict();
    
    #get the name of the file
    head,tail = ntpath.split(filename)
    
    #inputs
    dataset['NAME'] = tail;
    dataset['IN'] = [];
    dataset['OUT'] = [];
    
    with open(filename,'rb') as csvfile:
        data = csv.reader(csvfile, delimiter=delim,quoting=csv.QUOTE_NONE);
        
        #data
        for x in data:
                outl = x[:1];
                inl = x[1:];

                if inl != []:

                    dataset['IN'].append(inl);
                    dataset['OUT'].append(outl);
            
        
    for i,v in enumerate(dataset['IN']):
        
        if debug:
            print i, v;
            print [float(x) for x in v];
            
        #conver to float
        dataset['IN'][i] = [float(x) for x in v];
            

    for i,v in enumerate(dataset['OUT']):
        dataset['OUT'][i] = [float(x) for x in v];
        
    
    if verbose or debug:
        print dataset;

    return dataset;
#TEST
#data = read_csv_dataset('../datasets/australian-credit-data/australian_test.csv',',',1);
#print data;
