#Analysis


from pandas import DataFrame, read_csv;
import matplotlib.pyplot as plt;
import numpy as np;
import pickle;

#Drawing the graphs of control
#order is in form of
#Aussie CC, Heart, Diabetes, Cancer, Iris

Datasets = ['Card', 'Heart', 'Diabetes', 'Cancer', 'Iris' ];
NoinjectFiles = ['card.dat', 'heart.dat', 'diabetes.dat', 'cancer.dat', 'iris.dat'];
ControlFiles = ['Control/card.dat', 'Control/heart.dat', 'Control/diabetes.dat', 'Control/cancer.dat', 'Control/iris.dat'];
Control_results = {
    'test_err':  np.zeros((len(ControlFiles)))
   
};

NoInject_results = {
    
    'test_errs':  np.zeros((len(NoinjectFiles)))
};


opacity = 0.7;

#read the files
for i,file_name in enumerate(NoinjectFiles):
    #load the array from pickle
    mean_best_cost_noinject = pickle.load(open(file_name,"rb"));
    mean_best_cost_control = pickle.load(open(ControlFiles[i],"rb"));
    #plot 
    plt.plot(xrange(len(mean_best_cost_noinject)),mean_best_cost_noinject,
                    alpha = opacity,
                    color = 'r',
                    label = '${0} - no\ injection$'.format(Datasets[i]));

    plt.plot(xrange(len(mean_best_cost_control)),mean_best_cost_control,
                    alpha = opacity,
                    color = 'b',
                    label = '${0}  - with\ injection(random)$'.format(Datasets[i]));

    
    plt.ylabel('$Mean\ Training\ Error(MSE)$');
    plt.xlabel('$Generations(t)$');
    plt.title('$Convergence\ Comparison$');
    plt.legend();

    #plot and save
    plt.tight_layout();
    plt.savefig('errs_no_inject_convergence_'+Datasets[i]+'.png');
    plt.show();

