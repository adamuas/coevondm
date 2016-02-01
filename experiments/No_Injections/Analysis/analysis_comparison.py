#Analysis


from pandas import DataFrame, read_csv;
import matplotlib.pyplot as plt;
import numpy as np;

#Drawing the graphs of control
#order is in form of
#Aussie CC, Heart, Diabetes, Cancer, Iris

Datasets = ['$Card$', '$Heart$', '$Diabetes$', '$Cancer$', '$Iris$' ];
NoinjectFiles = ['card.csv', 'heart.csv', 'diabetes.csv', 'cancer.csv', 'iris.csv'];
ControlFiles = ['Control/Aussie CC.csv', 'Control/Heart.csv', 'Control/Diabetes.csv', 'Control/Cancer.csv', 'Control/Iris.csv'];
Control_results = {
    'train_errs': np.zeros((len(ControlFiles))),
    'test_errs':  np.zeros((len(ControlFiles))),
    'test_errs_std':  np.zeros((len(ControlFiles))),
    'train_errs_std': np.zeros((len(ControlFiles))),
    'conv_train_err':  np.zeros((len(ControlFiles))),
    'conv_test_err': np.zeros((len(ControlFiles))),
};

NoInject_results = {
    'train_errs': np.zeros((len(NoinjectFiles))),
    'test_errs':  np.zeros((len(NoinjectFiles))),
    'test_errs_std':  np.zeros((len(NoinjectFiles))),
    'train_errs_std': np.zeros((len(NoinjectFiles))),
    'conv_train_err':  np.zeros((len(NoinjectFiles))),
    'conv_test_err': np.zeros((len(NoinjectFiles))),
};




#read the files (No inject files)
for i,file_name in enumerate(NoinjectFiles):
    data = read_csv(file_name);
    test_err = data['avg_test_err'][0];
    train_err = data['avg_train_err'][0];
    test_err_std = data['std_test_err'][0];
    train_err_std = data['std_train_err '][0];
    #convg_test_err = data['conv_test_err'][0];
    #conv_train_err = data['conv_train_err'][0];

    NoInject_results['test_errs'][i]= test_err;
    NoInject_results['train_errs'][i]=train_err;
    NoInject_results['train_errs_std'][i] = train_err_std;
    NoInject_results['test_errs_std'][i] = test_err_std;
    #NoInject_results['conv_train_err'][i] = conv_train_err;
    #NoInject_results['conv_test_err'][i] = convg_test_err;

#read the files (Control files)
for i,file_name in enumerate(ControlFiles):
    data = read_csv(file_name);
    test_err = data['avg_test_err'][0];
    train_err = data['avg_train_err'][0];
    test_err_std = data['std_test_err'][0];
    train_err_std = data['std_train_err '][0];
    #convg_test_err = data['conv_test_err'][0];
    #conv_train_err = data['conv_train_err'][0];

    Control_results['test_errs'][i]= test_err;
    Control_results['train_errs'][i]=train_err;
    Control_results['train_errs_std'][i] = train_err_std;
    Control_results['test_errs_std'][i] = test_err_std;
    #Control_results['conv_train_err'][i] = conv_train_err;
    #Control_results['conv_test_err'][i] = convg_test_err;


#plot settings
NumDatasets = len(Datasets);
ind = np.arange(NumDatasets);
width = 0.23;
width2 = 0.23;
opacity = 0.7
error_config = {'ecolor': '0.3'}

def plot_errs():


    ## TRAIN ERROR
    #results = Control_results;

    fig, ax = plt.subplots()


    #results
    rects_control = plt.bar(ind,Control_results['train_errs'], width,
                  color = 'black',
                   alpha = opacity,
                   yerr =Control_results['train_errs_std']/np.sqrt(10),
                    label = r'$with\ injection(random)$');
    rects_control_ = plt.bar(ind+width2,NoInject_results['train_errs'], width,
                    color = 'gray',
                    alpha = opacity,
                    yerr =NoInject_results['train_errs_std']/np.sqrt(10),
                    label = r'$no\ injection$');

    
    plt.ylabel('$Training\ Error(MSE)$');
    plt.title(r'$Transfer\ Functions\ Injection - Performance\  Comparison$');
    plt.xticks(ind+width, Datasets);
    plt.legend();

    #plot and save
    plt.tight_layout();
    plt.savefig('errs_comp_inject_train'+'.png');
    plt.show();


    # TEST ERROR

    fig, ax = plt.subplots()


    #results
    rects_control = plt.bar(ind,Control_results['test_errs'], width,
                  color = 'black',
                   alpha = opacity,
                   yerr =Control_results['test_errs_std']/np.sqrt(10),
                    label = r'$with\ injection(random)$');
    rects_control_ = plt.bar(ind+width2,NoInject_results['test_errs'], width,
                    color = 'gray',
                    alpha = opacity,
                    yerr =NoInject_results['test_errs_std']/np.sqrt(10),
                    label = r'$no\ injection$');

    
    plt.ylabel('$Test\ Error(MSE)$');
    plt.title(r'$Transfer\ Functions\ Injection - Performance\  Comparison $');
    plt.xticks(ind+width, Datasets);
    plt.legend();

    #plot and save
    plt.tight_layout();
    plt.savefig('errs_comp_inject_test'+'.png');
    plt.show();



def plot_errsh():


    results = Control_results;

    fig, ax = plt.subplots()


    #results
    rects_train = plt.barh(ind,results['train_errs'], width,
                    color = 'b',
                    alpha = opacity,
                    xerr =results['train_errs_std']/np.sqrt(10),
                    label = '$train$');
    rects_test = plt.barh(ind+width,results['test_errs'], width,
                    color = 'r',
                    alpha = opacity,
                    xerr =results['test_errs_std']/np.sqrt(10),
                    label = 'test');

    
    plt.ylabel('Performance (Error)');
    plt.title('Error (MSE)')
    plt.yticks(ind+width, Datasets);
    plt.legend();

    #plot and save
    plt.tight_layout();
    plt.savefig('errs'+'.png');
    plt.show();


def plot_err_comp():


    results = Control_results;

    fig, ax = plt.subplots()


    #results
    rects_train = plt.bar(ind,results['train_errs'], width,
                    color = 'b',
                    alpha = opacity,
                    yerr =results['train_errs_std'],
                    label = 'train');
    rects_test = plt.bar(ind+width,results['test_errs'], width,
                    color = 'r',
                    alpha = opacity,
                    yerr =results['test_errs_std'],
                    label = 'test');

    plt.xlabel('Datasets');
    plt.ylabel('Error(MSE)');
    plt.title('Performance (Error)')
    plt.xticks(ind+width, Datasets);
    plt.legend();

    #plot and save
    plt.tight_layout();
    plt.savefig('errs'+'.png');
    plt.show();

def plot_comparison_conv():
    """ plots the comparison between """


    fig, ax = plt.subplots()

    #results
    rects_control = plt.bar(ind,Control_results['conv_test_err'], width,
                    color = 'b',
                    alpha = opacity,
                    label = 'Control');

    plt.xlabel('Datasets');
    plt.ylabel('Error(MSE)');
    plt.title('Convergence Comparison')
    plt.xticks(ind+width, Datasets);
    plt.legend();

    #plot and save
    #plt.tight_layout();
    plt.savefig('conv_'+'Comparison_BagControl'+'.png');
    plt.show();






#run
plot_errs();

