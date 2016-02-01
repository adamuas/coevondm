import csv,sys;

def openForWrite(filename, mode = 'wb'):
    """ returns a file object"""
    f = open(filename, mode);
    
    return f;

def writeHeadToFile(fileObj,data):
    """ Write the header to the file """
    w = csv.DictWriter(f,sorted(data.keys()));
    w.writeheader();
    
def writeRowToFile(fileObj,row_data):
    """ Writes a row to the file"""
    w.writerow({k:v for k,v in results.items()});
    f.close();
    
def write(filename,results,mode='wb'):
    """ takes in the values of the results in a dictionary for writting. """
    f = open(filename,mode)
    w = csv.DictWriter(f,sorted(results.keys()))
    w.writeheader()
    w.writerow({k:v for k,v in results.items()})
    f.close();

    
    
#TEST
#data = dict();
#data['var'] = 0.45;
#data['mean'] = 0.45;
#data['weights'] = [0.3,0.3,0.3];
#
#write('test.csv',data);
    
    