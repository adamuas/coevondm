import sys;
sys.path.insert(0, "../datasets");
sys.path.insert(0, "../tools");

import ConfigParser;
import csv;
import numpy as np;
import artificial_dataset;
import os;
import proben1;
import csvReader;
import iris;
import sonar;
import hepatitis;
import abalone;
import lenses;
import parkinsons;
import echocardiogram;
import vertebral_column;
import seeds;
import acute_inflamation;
import ionosphere;
import bankruptcy;
import monks;
import spect_heart;
import lung_cancer;

#get the path of proben1
fdir = os.path.join(proben1.__path__.pop());
fdir_iris = os.path.join(iris.__path__.pop());
fdir_sonar = os.path.join(sonar.__path__.pop());
fdir_hepatitis = os.path.join(hepatitis.__path__.pop());
fdir_abalone = os.path.join(abalone.__path__.pop());
fdir_lenses = os.path.join(lenses.__path__.pop());
fdir_parkinsons = os.path.join(parkinsons.__path__.pop());
fdir_echocardiogram = os.path.join(echocardiogram.__path__.pop());
fdir_seeds = os.path.join(seeds.__path__.pop());
fdir_vertebral_column = os.path.join(vertebral_column.__path__.pop());
fdir_monks = os.path.join(monks.__path__.pop());
fdir_ionosphere = os.path.join(ionosphere.__path__.pop());
fdir_inflamations = os.path.join(acute_inflamation.__path__.pop());
fdir_bakruptcy = os.path.join(bankruptcy.__path__.pop());
fdir_spect_heart = os.path.join(spect_heart.__path__.pop());
fdir_lung_cancer = os.path.join(lung_cancer.__path__.pop());


#PROBEN1 SET
proben1_set = ['cancer','diabetes', 'glass', 'thyroid', 'card', 'heart', 'flare', 'horse'];

#directories for the bechmarks
bechmark_dir = {
    'cancer': (os.path.join(fdir,'./cancer/cancer1.csv'),#Include in big benchmark
               os.path.join(fdir,'./cancer/cancer2.csv'),
               os.path.join(fdir,'./cancer/cancer3.csv')),
    
    'diabetes': (os.path.join(fdir,'./diabetes/diabetes1.csv'),#Include in big benchmark
                 os.path.join(fdir,'./diabetes/diabetes2.csv'),
                 os.path.join(fdir,'./diabetes/diabetes3.csv')),
    
    'mushroom': (os.path.join(fdir,'/mushroom/mushroom1.csv'),
                 os.path.join(fdir,'/mushroom/mushroom2.csv'),
                 os.path.join(fdir,'/mushroom/mushroom3.csv')),

    
    'card': (os.path.join(fdir,'./card/card1.csv'),#Include in big benchmark
             os.path.join(fdir,'./card/card2.csv'),
             os.path.join(fdir,'./card/card3.csv')),
    
    'thyroid': (os.path.join(fdir,'./thyroid/thyroid1.csv'), #EXCLUDED
                os.path.join(fdir,'./thyroid/thyroid2.csv'),#EXCLUDED
                os.path.join(fdir,'./thyroid/thyroid3.csv')),#EXCLUDED

    'building': (os.path.join(fdir,'./building/building1.csv'), #EXCLUDED
                os.path.join(fdir,'./building/building2.csv'),#EXCLUDED
                os.path.join(fdir,'./building/building3.csv')),#EXCLUDED

    'flare': (os.path.join(fdir,'flare/flare1.csv'), #Include in big benchmark
              os.path.join(fdir,'flare/flare2.csv'),
              os.path.join(fdir,'flare/flare3.csv')),

    'glass': (os.path.join(fdir,'glass/glass1.csv'), #include in biig benchmark
              os.path.join(fdir,'glass/glass2.csv'),
              os.path.join(fdir,'glass/glass3.csv')),
    
    'heart': (os.path.join(fdir,'./heart/heart1.csv'), #Include in big benchmark
              os.path.join(fdir,'./heart/heart2.csv'),
              os.path.join(fdir,'./heart/heart3.csv')),

    'horse': (os.path.join(fdir,'./horse/horse1.csv'),#Include in big benchmark
              os.path.join(fdir,'./horse/horse2.csv'),
              os.path.join(fdir,'./horse/heart3.csv')),

    
    
    #Lab Benchmark
    'iris': ('./iris_train_trioutput.csv',
             './iris_test_trioutput.csv',
             './iris_all_2output.csv'),
    'sonar': ('./sonar/sonar_train.csv',
              './sonar/sonar_test.csv'),
    'hepatitis': ('./hepatitis/hepatitis_all.csv',
              './hepatitis/hepatitis_test.csv'),

    'abalone': ('./abalone/abalone_all.csv',
              './abalone/abalone_test.csv'),
    'lenses': ('./lenses/lenses.csv',
              '.'),


    'xor': ('./',
            './'),
    'double_spiral': ('./',
                      './'),
    'n_parity': ('./',
                 './'
                )
    };

bechmark_config_dir = {
    'card': os.path.join(fdir,'./card/config.cfg'),
    'cancer': os.path.join(fdir,'./cancer/config.cfg'),
    'diabetes': os.path.join(fdir,'./diabetes/config.cfg'),
    'mushroom': os.path.join(fdir,'./mushroom/config.cfg'),
    'thyroid': os.path.join(fdir,'./thyroid/config.cfg'),
    'flare': os.path.join(fdir,'./flare/config.cfg'),
    'horse': os.path.join(fdir,'./horse/config.cfg'),
    'building': os.path.join(fdir,'./building/config.cfg'),
    'heart': os.path.join(fdir,'./heart/config.cfg'),
    'glass': os.path.join(fdir,'./glass/config.cfg'),
    'segmentation': './segmentation/config.cfg',
    #Lab Benchmark
    'iris': './iris/config.cfg',
    'sonar': './sonar/config.cfg',
    'hepatitis': './hepatitis/config.cfg',

    'xor': './',
    'double_spiral':'./',
    'n_parity':'./'
    };




def clean(lst):
    
    nwlst = [];
    for i in lst:
        if i != '':
            nwlst.append(i);
            
    return nwlst;
            

def load_file(key):
    #Load the file
    if bechmark_dir.has_key(key):
        #load train, validation and test set
        return (open(bechmark_dir[key][0]),open(bechmark_dir[key][1]),open(bechmark_dir[key][2]));
        
def read_settings(key):
    #load the configuration file
    
    bench_config = {
        'bool_in' : 0,
        'real_in' : 0,
        'bool_out': 0,
        'real_out' : 0,
        'train': 0,
        'validation': 0,
        'test': 0
    };
    
    config = ConfigParser.ConfigParser();
    config.readfp(open(bechmark_config_dir[key]));
    
    bench_config['bool_in']  = config.getint('meta-info','bool_in');
    bench_config['real_in']  = config.getint('meta-info','real_in');
    bench_config['bool_out']  = config.getint('meta-info','bool_out');
    bench_config['real_out']  = config.getint('meta-info','real_out');
    bench_config['train']  = config.getint('meta-info','training_examples');
    bench_config['validation']  = config.getint('meta-info','validation_examples');
    bench_config['test']  = config.getint('meta-info','test_examples');
    
    return bench_config;

def process_file(f,key,singleOutput = True):
    #processes the file and returns the datasets (i.e. trainset, testset, and validation set).
    
    dataset = {'IN':[], 'OUT' : [], 'INFO': []}
    
    
    reader = csv.reader(f,delimiter=' ');

    bench_config = read_settings(key);
    
    #get the number of inputs
    num_inputs = bench_config['bool_in'] + bench_config['real_in'];
    num_outputs = bench_config['bool_out'] + bench_config['real_out'];
            
    dataset['INFO'] = {'num_inputs': num_inputs, 'num_outputs': num_outputs};
    
    print "num_in:",num_inputs;
    print "num_out:",num_outputs;
    

    
    for row in reader:
        row  = clean(row);

        if row == []:
            #skip
            continue;

        example  = [float(r) for r in row];
        example_in = example[:num_inputs];
        example_out = example[num_inputs:];
        dataset['IN'].append(example_in);
        
        if len(example_out) == 2 and singleOutput == True:
            
            if example_out[1]:
                dataset['OUT'].append([0.1]);
            else:
                dataset['OUT'].append([0.9]);
        else:
            print ">> PROBEN: Check the number of output nodes, I was unable to squash the number of output nodes for you.";
            dataset['OUT'].append(example_out);
                
    
    return dataset;
    
    

def load_benchmark(key, singleOutput = True, writeToFile= False):
    #load dataset
    
    if key in proben1_set:
        f_train, f_vald, f_test = load_file(key);
        dataset_train = process_file(f_train,key,singleOutput);
        dataset_valid = process_file(f_vald,key,singleOutput);
        dataset_test = process_file(f_test,key,singleOutput);
        
    
        
    if writeToFile:
        writeBenchmark(dataset_train,key,'train');
        writeBenchmark(dataset_valid,key,'valid');
        writeBenchmark(dataset_test,key,'test');
        
        
    
    return {'name': key, 'train': dataset_train, 'validation': dataset_valid, 'test': dataset_test};
    
def writeBenchmark(dataset,key,part_type= 'train'):
    #writes the datasets
    
    
    writer = csv.writer(open(key+'_'+part_type+'_in'+'.csv', 'wb'))
    for value in dataset['IN']:
       writer.writerow(value)
       
    writer = csv.writer(open(key+'_'+part_type+'_out'+'.csv', 'wb'))
    for value in dataset['OUT']:
       writer.writerow(value)
       



#TEST


class lab_bencmark:
    """
    This consists of datasets that facilitate quick experimentation for getting results
    """
    
    def iris(self):

        #iris dataset
        num_params_in = 4;
        iris_all = csvReader.read_csv_dataset_right(os.path.join(fdir_iris,'iris_all.csv'),',',num_params_in);
        iris_train = csvReader.read_csv_dataset_right(os.path.join(fdir_iris,'iris_train1.csv'),',',num_params_in);
        iris_test = csvReader.read_csv_dataset_right(os.path.join(fdir_iris,'iris_test1.csv'),',',num_params_in);

        return {'NAME': 'IRIS', 'train': iris_train, 'test': iris_test, 'all': iris_all};
    
    def sonar(self):
        #sonar dataset
        num_params_in = 60;
        sonar_train = csvReader.read_csv_dataset_right(os.path.join(fdir_sonar,'sonar_train.csv'),',',num_params_in);
        sonar_test = csvReader.read_csv_dataset_right(os.path.join(fdir_sonar,'sonar_test.csv'),',',num_params_in);
        sonar_all = csvReader.read_csv_dataset_right(os.path.join(fdir_sonar,'sonar_all.csv'),',',num_params_in);

        return {'NAME': 'SONAR', 'train': sonar_train, 'test': sonar_test, 'all':sonar_all};

    def abalone(self):

        #abalone
        num_params_in = 8;
        abalone_all = csvReader.read_csv_dataset_left(os.path.join(fdir_abalone,'abalone_all.csv'),',');
        abalone_train = csvReader.read_csv_dataset_left(os.path.join(fdir_abalone,'abalone_train.csv'),',');
        abalone_test = csvReader.read_csv_dataset_left(os.path.join(fdir_abalone,'abalone_test.csv'),',');

        return {'NAME': 'Abalone', 'train': abalone_train, 'test': abalone_test, 'all':abalone_all};

    def lenses(self):

        #lenses
        num_params_in = 4;
        lenses_train = csvReader.read_csv_dataset_left(os.path.join(fdir_lenses,'lenses_train.csv'),',');
        lenses_test = csvReader.read_csv_dataset_left(os.path.join(fdir_lenses,'lenses_test.csv'),',');
        lenses_all = csvReader.read_csv_dataset_left(os.path.join(fdir_lenses,'lenses.csv'),',');

        return {'NAME': 'Lenses', 'train': lenses_train, 'test': lenses_test, 'all':lenses_all};

    def parkinsons(self):

        #parkinsons
        num_params_in = 22;
        parkinsons_train = csvReader.read_csv_dataset_right(os.path.join(fdir_parkinsons,'parkinsons_train.csv'),',',num_params_in);
        parkinsons_test = csvReader.read_csv_dataset_right(os.path.join(fdir_parkinsons,'parkinsons_test.csv'),',',num_params_in);
        parkinsons_all = csvReader.read_csv_dataset_right(os.path.join(fdir_parkinsons,'parkinsons_all.csv'),',',num_params_in);

        return {'NAME': 'Parkinsons', 'train': parkinsons_train, 'test': parkinsons_test, 'all':parkinsons_all};

    def echocardiogram(self):

        #echodiagram
        num_params_in = 11;
        echocardiogram_train = csvReader.read_csv_dataset_left(os.path.join(fdir_echocardiogram,'echocardiogram_train.csv'),',');
        echocardiogram_test = csvReader.read_csv_dataset_left(os.path.join(fdir_echocardiogram,'echocardiogram_test.csv'),',');
        echocardiogram_all = csvReader.read_csv_dataset_left(os.path.join(fdir_echocardiogram,'echocardiogram_all.csv'),',');

        return {'NAME': 'Echocardiogram', 'train': echocardiogram_train, 'test': echocardiogram_test, 'all':echocardiogram_all};


    def seeds(self):

        #seeds
        num_params_in = 7;
        seeds_train = csvReader.read_csv_dataset_right(os.path.join(fdir_seeds,'seeds_train.csv'),',', num_params_in);
        seeds_test = csvReader.read_csv_dataset_right(os.path.join(fdir_seeds,'seeds_test.csv'),',', num_params_in);
        seeds_all = csvReader.read_csv_dataset_right(os.path.join(fdir_seeds,'seeds_all.csv'),',', num_params_in);

        return {'NAME': 'Seeds', 'train': seeds_train, 'test': seeds_test, 'all':seeds_all};

    def vertebral_column2C(self):

        #vertebral column with two classes
        num_params_in = 6;
        column2c_all = csvReader.read_csv_dataset_right(os.path.join(fdir_vertebral_column,'column_2C_all.csv'),',',num_params_in);
        column2c_train = csvReader.read_csv_dataset_right(os.path.join(fdir_vertebral_column,'column_2C_train.csv'),',',num_params_in);
        column2c_test = csvReader.read_csv_dataset_right(os.path.join(fdir_vertebral_column,'column_2C_test.csv'),',',num_params_in);

        return {'NAME': 'VertebralCol2C', 'train': column2c_train, 'test': column2c_test, 'all':column2c_all};

    def vertebral_column3C(self):

        #vertebral column with three classes
        num_params_in = 6;
        column3c_all = csvReader.read_csv_dataset_right(os.path.join(fdir_vertebral_column,'column_3C_all.csv'),',',num_params_in);
        column3c_train = csvReader.read_csv_dataset_right(os.path.join(fdir_vertebral_column,'column_3C_train.csv'),',',num_params_in);
        column3c_test = csvReader.read_csv_dataset_right(os.path.join(fdir_vertebral_column,'column_3C_test.csv'),',',num_params_in);

        return {'NAME': 'VertebralCol3C', 'train': column3c_train, 'test': column3c_test, 'all':column3c_all};


    def hepatitis(self):

        #hepatitis
        num_params_in = 19;
        hepatitis_all = csvReader.read_csv_dataset_left(os.path.join(fdir_hepatitis,'hepatitis_all.csv'),',');
        hepatitis_train = csvReader.read_csv_dataset_left(os.path.join(fdir_hepatitis,'hepatitis_train.csv'),',');
        hepatitis_test = csvReader.read_csv_dataset_left(os.path.join(fdir_hepatitis,'hepatitis_test.csv'),',');

        return {'NAME': 'Hepatitis', 'train': hepatitis_train, 'test': hepatitis_test, 'all':hepatitis_all};

    def spect_heart(self):

        spect_heart_all = csvReader.read_csv_dataset_left(os.path.join(fdir_spect_heart, 'spect_heart_all.csv'), ',');
        spect_heart_train = csvReader.read_csv_dataset_left(os.path.join(fdir_spect_heart, 'spect_heart_train.csv'), ',');
        spect_heart_test = csvReader.read_csv_dataset_left(os.path.join(fdir_spect_heart, 'spect_heart_test.csv'), ',');

        return {'NAME': 'Spect_Heart', 'train': spect_heart_train, 'test': spect_heart_test, 'all':spect_heart_all};

    def lung_cancer(self):

        #lung_cancer_all = csvReader.read_csv_dataset_left(os.path.join(fdir_lung_cancer, 'lung_cancer_all.csv'), ',');
        lung_cancer_train = csvReader.read_csv_dataset_left(os.path.join(fdir_lung_cancer, 'lung_cancer_train.csv'),',');
        lung_cancer_test = csvReader.read_csv_dataset_left(os.path.join(fdir_lung_cancer, 'lung_cancer_test.csv'),',');

        return {'NAME': 'Lung_Cancer', 'train': lung_cancer_train, 'test': lung_cancer_test, 'all':None};

    def monks1(self):


        monks_train = csvReader.read_csv_dataset_left(os.path.join(fdir_monks, 'monks1_train.csv'),',');
        monks_test = csvReader.read_csv_dataset_left(os.path.join(fdir_monks, 'monks1_test.csv'),',');

        return {'NAME': 'Monks1', 'train': monks_train, 'test': monks_test, 'all':None};

    def monks2(self):

        monks_train = csvReader.read_csv_dataset_left(os.path.join(fdir_monks, 'monks2_train.csv'),',');
        monks_test = csvReader.read_csv_dataset_left(os.path.join(fdir_monks, 'monks2_test.csv'),',');

        return {'NAME': 'Monks2', 'train': monks_train, 'test': monks_test, 'all':None};

    def monks3(self):
        monks_train = csvReader.read_csv_dataset_left(os.path.join(fdir_monks, 'monks3_train.csv'),',');
        monks_test = csvReader.read_csv_dataset_left(os.path.join(fdir_monks, 'monks3_test.csv'),',');

        return {'NAME': 'Monks3', 'train': monks_train, 'test': monks_test, 'all':None};

    def bankruptcy(self):

        num_param_in = 6;
        bankruptcy_all = csvReader.read_csv_dataset_right(os.path.join(fdir_bakruptcy, 'bankruptcy_all.csv'), ',',num_param_in);
        bankruptcy_train = csvReader.read_csv_dataset_right(os.path.join(fdir_bakruptcy, 'bankruptcy_train.csv'), ',',num_param_in);
        bankruptcy_test = csvReader.read_csv_dataset_right(os.path.join(fdir_bakruptcy, 'bankruptcy_test.csv'), ',',num_param_in);

        return {'NAME': 'Bankruptcy', 'train': bankruptcy_train, 'test': bankruptcy_test, 'all':bankruptcy_all};

    def ionosphere(self):
        num_param_in = 34;
        ionosphere_all = csvReader.read_csv_dataset_right(os.path.join(fdir_ionosphere, 'ionosphere_all.csv'), ',',num_param_in);
        ionosphere_train = csvReader.read_csv_dataset_right(os.path.join(fdir_ionosphere, 'ionosphere_train.csv'), ',',num_param_in);
        ionosphere_test = csvReader.read_csv_dataset_right(os.path.join(fdir_ionosphere, 'ionosphere_test.csv'), ',',num_param_in);

        return {'NAME': 'Ionosphere', 'train': ionosphere_train, 'test': ionosphere_test, 'all':ionosphere_all};

    def acute_inflamation(self):
        num_param_in = 6;
        acute_inflamation_all = csvReader.read_csv_dataset_right(os.path.join(fdir_inflamations, 'inflamation_all.csv'), ',',num_param_in);
        acute_inflamation_train = csvReader.read_csv_dataset_right(os.path.join(fdir_inflamations, 'inflamation_train.csv'), ',',num_param_in);
        acute_inflamation_test = csvReader.read_csv_dataset_right(os.path.join(fdir_inflamations, 'inflamation_test.csv'), ',',num_param_in);

        return {'NAME': 'inflamation', 'train': acute_inflamation_train, 'test': acute_inflamation_test, 'all':acute_inflamation_all};



    
    def xor(self,n):
        D = artificial_dataset.Xor(n);
        num_training = int(0.7 * n);
        train_set = {
            'IN': D['IN'][:num_training],
            'OUT': D['OUT'][:num_training]
        };
        test_set = {
            'IN': D['IN'][num_training:],
            'OUT': D['OUT'][num_training:]
        };

        return {'NAME': 'XOR', 'train': train_set, 'test': test_set};
    
    def double_spiral(self):
        return
    
    def n_parity(self,k,n):
        return artificial_dataset.Parity(k,n);
    

class proben1_bechmark:
    """
    This is the proben1 bechmark and consist of datasets for validation of experimental results from the lab_benchmark.
    """
    def __init__(self,singleOutput = True):
        self.singleOutput = singleOutput;
    
    def breast_cancer(self):
        #returns the breast cancer dataset
        return load_benchmark('cancer',self.singleOutput);
    
    def australian_cc(self):
        #returns the australian dataset
        return load_benchmark('card',self.singleOutput);
    
    def diabetes(self):
        return load_benchmark('diabetes',self.singleOutput);
    
    def splice_junction(self):
        return None;
    
    def glass(self):
        return load_benchmark('glass', self.singleOutput);
    
    def heart(self):
        return load_benchmark('heart',self.singleOutput);
    
    def horse(self):
        return load_benchmark('horse');
    
    def mushroom(self): #EXCLUDED
        return load_benchmark('mushroom',self.singleOutput);
    
    def soybean(self): #EXCLUDED
        pass;
    
    def thyroid(self): #EXCLUDED
        return load_benchmark('thyroid');
    
    def building(self):#EXCLUDED
        return load_benchmark('building');
    
    def flare(self):
        return load_benchmark('flare');
    
    def hearta(self):
        pass;

    def heartac(self):
        pass;
    
        
class zheng_bechmark:
    """
        This is the Zheng 1993 Bechmark and consists of datasets for the validation of experimental results from the lab_benchmark.
        Unlike the Proben1, this dataset consist of a variety of different bechmarks and overlaps with proben1.
    """
    
    def breast_cancer(self):
        #returns the breast cancer dataset
        
        pass;
    
    def diabetes(self):
        #returns the diabetes dataset
        pass;
    def hepatitis(self):
        pass;
    
    def LED7(self):
        pass;
    
    def LED24(self):
        pass;
    
    def lymphography(self):
        pass;
    
    def monks2(self):
        pass;
    
    def mushroom(self):
        pass;
    
    def nettalk(self):
        pass;
    
    def promoter(self):
        pass;
    
    def soybean(self):
        pass;
    
    def thyroid(self):
        pass;
    
    def waveform40(self):
        pass;
    
    
##print sys.path;
##p = proben1_bechmark();
##p.breast_cancer();
##
#
#
# #
# # # # #
# lb = lab_bencmark();
# print lb.sonar()['train']['OUT'];
# print len(lb.sonar()['train']['OUT'])
#
#


