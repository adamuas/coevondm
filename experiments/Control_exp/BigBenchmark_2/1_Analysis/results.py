import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

## the data

labels = ('DecisionTree', 'KNN', 'SVM-RBF','SVM-LINEAR','SVM-POLY','NB-GAUSS','NB-BERNOULLI',  'CoevoNDM', 'MLP', 'RBF' ); #'MLP',
datasets = ['iris', 'sonar', 'abalone', 'lenses', 'parkinsons'];
datasets2 = ['vertebral2C', 'vertebral3C', 'Hepatitis', 'Echocardiogram', 'Lung Cancer']
datasets3 =  ['SPECT Heart', 'Seeds', 'Monks1','Monks2','Monks3']
datasets4 = ['Bankruptcy','Ionosphere','Acute Inflamations', 'card', 'cancer', 'diabetes', 'heart']
N = len(datasets);

#NAIVE BAYES
GAUSS_NBMean = [0.009450,0.080000, 0.302485, 0.240000, 0.090900];
GAUSS_NBMean2 = [0.009450,0.080000, 0.302485, 0.240000, 0.090900];
GAUSS_NBMean3 = [0.009450,0.080000, 0.302485, 0.240000, 0.090900];
GAUSS_NBMean4 = [0.009450,0.080000, 0.302485, 0.240000, 0.090900];


#DECISION TREE
DCTMean = [0.046667, 1.235714, 0.520455,0.456000,0.060000 ];
DCTMean2 = [];
DCTMean3 =  [];
DCTMean4 =  [];

#NEAREST NEIGHBOR
KNNMean = [0.006750, 0.168000, 0.353780 , 0.202667 ,0.040500];
KNNMean2 = [0.85, 0.97, 0.73, 0.80];
KNNMean3= [0.85, 0.97, 0.73, 0.80];
KNNMean4= [0.85, 0.97, 0.73, 0.80];

#SVM SIG
SIGSVMEAN = [0.272633,0.394286, 0.467803, 0.346667, 0.198000 ];
SIGSVMEAN2 = [];
SIGSVMEAN3= [];
SIGSVMEAN4 = [];


#RBF
RBFMEAN = [0.100278649786, 0.361098791857,0.306086480524 , 0.193454483259, 0.609587628866];

RBFSVMMean = [0.002700,0.196571, 0.386038, 0.325333, 0.150300];
RBFSVMMean2 = [];
RBFSVMMean3= [];
RBFSVMMean4 = [];



#NEURAL NETWORK
MLPMean = [0.340833333333, 0.398888888889, 0.469026548673, 0.391538461538, 0.307083333333];
MLPMean2 = 0.406116504854, 0.347108490566, 0.0861538461538, 0.64, 0.307];
MLPMean3 = [0.41, 0.195577464789, 0.505, 0.625088757396, 0.513032786885];
MLPMean4 = [0.506124031008, 0.40771, 0.40997, 0.4540, 0.5308, 0.2858, 0.452];

## necessary variables
ind = np.arange(N);                # the x locations for the groups
width = 0.10;                   # the width of the bars

## the bars

### NDM MEANS
#ORDER : ['iris', 'sonar', 'abalone', 'lenses', 'parkinsons'];
NDMMean = [0.025026287, 0.214389465, 0.169150169,0.174057842333, 0.1638901];
NDMMean2 = [ 0.194150027, 0.087200956, 0.204477766,0.060090766,0.1100013755];
NDMMean3 = [0.107270098,0.054481567,0.19198929,0.167690503,0.119573707  ];
NDMMean4 = [0.013084332,0.102755169,0.256595248, 0.111027958, 0.024285554,0.122745779,0.124378078]; #no 2 is ionosphere;


##################################### FIRST GRAPH ####################################################################


##DECISION TREE
DCT = ax.barh(ind, DCTMean, width, \
                color='purple');
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))

###KNN
KNN = ax.barh(ind+width*1, KNNMean, width,\
                    color='red');
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))



#SVM
SVMSIG = ax.barh(ind+(width*2), SIGSVMEAN, width,\
                    color='black')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#SVM
SVMRBF = ax.barh(ind+(width*3), RBFSVMMean, width,\
                    color='gray')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean, width,\
                    color='yellow')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean, width,\
                    color='orange')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))


#RBF
RBF = ax.barh(ind+(width*5), RBFMEAN, width,\
                    color='blue')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))
#Naive Bayes
NB_GAUSS = ax.barh(ind+(width*6), GAUSS_NBMean, width,\
                color='green')
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))


NDM = ax.barh(ind+(width*7), NDMMean, width\
                    color='cyan')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))


##################################### SECOND GRAPH ####################################################################


##DECISION TREE
DCT = ax.barh(ind, DCTMean2, width, \
                color='purple');
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))

###KNN
KNN = ax.barh(ind+width*1, KNNMean2, width,\
                    color='red');
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))



#SVM
SVMSIG = ax.barh(ind+(width*2), SIGSVMEAN2, width,\
                    color='black')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#SVM
SVMRBF = ax.barh(ind+(width*3), RBFSVMMean2, width,\
                    color='gray')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean2, width,\
                    color='yellow')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean2, width,\
                    color='orange')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))


#RBF
RBF = ax.barh(ind+(width*5), RBFMEAN2, width,\
                    color='blue')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))
#Naive Bayes
NB_GAUSS = ax.barh(ind+(width*6), GAUSS_NBMean2, width,\
                color='green')
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))


NDM = ax.barh(ind+(width*7), NDMMean2, width\
                    color='cyan')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))



##################################### THIRD GRAPH ####################################################################


##DECISION TREE
DCT = ax.barh(ind, DCTMean3, width, \
                color='purple');
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))

###KNN
KNN = ax.barh(ind+width*1, KNNMean3, width,\
                    color='red');
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))



#SVM
SVMSIG = ax.barh(ind+(width*2), SIGSVMEAN3, width,\
                    color='black')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#SVM
SVMRBF = ax.barh(ind+(width*3), RBFSVMMean3, width,\
                    color='gray')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean3, width,\
                    color='yellow')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean3, width,\
                    color='orange')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))


#RBF
RBF = ax.barh(ind+(width*5), RBFMEAN3, width,\
                    color='blue')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))
#Naive Bayes
NB_GAUSS = ax.barh(ind+(width*6), GAUSS_NBMean3, width,\
                color='green')
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))


NDM = ax.barh(ind+(width*7), NDMMean3, width, \
                    color='cyan')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))




##################################### FOUR GRAPH ####################################################################


##DECISION TREE
DCT = ax.barh(ind, DCTMean4, width, \
                color='purple');
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))

###KNN
KNN = ax.barh(ind+width*1, KNNMean4, width,\
                    color='red');
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))



#SVM
SVMSIG = ax.barh(ind+(width*2), SIGSVMEAN4, width,\
                    color='black')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#SVM
SVMRBF = ax.barh(ind+(width*3), RBFSVMMean4, width,\
                    color='gray')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean4, width,\
                    color='yellow')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#MLP
MLP = ax.barh(ind+(width*4), MLPMean4, width,\
                    color='orange')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))


#RBF
RBF = ax.barh(ind+(width*5), RBFMEAN4, width,\
                    color='blue')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))
#Naive Bayes
NB_GAUSS = ax.barh(ind+(width*6), GAUSS_NBMean4, width,\
                color='green')
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))


NDM = ax.barh(ind+(width*7), NDMMean4, width, \
                    color='cyan')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))






# axes and labels
ax.set_ylim(-width,len(ind)+width);
ax.set_xlim(0.001, 1.2);
##ax.set_yscale(u'log');
ax.set_xlabel(r'$Error(MSE)$');
ax.set_ylabel(r'$Dataset$');
ax.set_title(r'$How\ does\ it\ compare?$');
yTickMarks = datasets; 
ax.set_yticks(ind+width);
ytickNames = ax.set_yticklabels(yTickMarks);
plt.setp(ytickNames, rotation=45, fontsize=10);

## add a legend
#ax.legend(MLP[0], labels) # (DCT[0], KNN[0], RBFSVM[0], LINEARSVM[0], POLYSVM[0],GAUSS_NB[0],BERNOUL_NB[0], NDM[0]), labels ) 

plt.show()
