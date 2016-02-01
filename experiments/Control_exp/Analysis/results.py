import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

## the data
N = 4
labels = ('DecisionTree', 'KNN', 'SVM-RBF','SVM-LINEAR','SVM-POLY','NB-GAUSS','NB-BERNOULLI',  'CoevoNDM' ); #'MLP',
datasets = ['card', 'cancer', 'diabetes', 'heart'];

#SVM
RBFSVMMean = [0.865217,0.969957, 0.777344, 0.838043 ];
LINEARSVMMean = [ 0.82,0.97,  0.50,0.65 ];
POLYSVMMean = [ 0.68,0.94, 0.37, 0.63 ];

#NAIVE BAYES
BERNOULLI_NBMean = [0.81, 0.66,  0.64, 0.82];
GAUSS_NBMean = [0.67,0.96, 0.75, 0.70];

#DECISION TREE
DCTMean = [0.84, 0.94, 0.70, 0.73];
DCTStd =   [0.08, 0.07, 0.10, 0.04];

#NEAREST NEIGHBOR
KNNMean = [0.85, 0.97, 0.73, 0.80];
KNNStd =   [0.08, 0.04, 0.13, 0.06];

#NEURAL NETWORK
MLPMean = [0.0, 0.86817, 0.0, 0.0];
NDMMean = [0.8951, 0.9703, 0.8829, 0.8811];
## necessary variables
ind = np.arange(N)                # the x locations for the groups
width = 0.07                      # the width of the bars

## the bars
#DECISION TREE
DCT = ax.barh(ind, DCTMean, width,
                color='purple')
                #yerr=DCTStd,
                #error_kw=dict(elinewidth=2,ecolor='red'))

#KNN
KNN = ax.barh(ind+width*1, KNNMean, width,
                    color='red')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))



#SVM

RBFSVM = ax.barh(ind+(width*2), RBFSVMMean, width,
                    color='black')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))


LINEARSVM = ax.barh(ind+(width*3), LINEARSVMMean, width,
                    color='gray')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

POLYSVM = ax.barh(ind+(width*4), POLYSVMMean, width,
                    color='white')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#NAIVE BAYES
GAUSS_NB = ax.barh(ind+(width*5), GAUSS_NBMean, width,
                    color='green')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))
BERNOUL_NB = ax.barh(ind+(width*6), BERNOULLI_NBMean, width,
                    color='orange')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

#NEURAL NETWORKS
#MLP = ax.barh(ind+(width*7), MLPMean, width, color = 'yellow');
NDM = ax.barh(ind+(width*8), NDMMean, width,
                    color='blue')
                    #yerr=KNNStd,
                    #error_kw=dict(elinewidth=2,ecolor='black'))

# axes and labels
ax.set_ylim(-width,len(ind)+width)
ax.set_xlim(0.1, 1.2)
##ax.set_yscale(u'log');
ax.set_xlabel(r'$Accuracy$')
ax.set_ylabel(r'$Dataset$')
ax.set_title(r'$How\ does\ it\ compare?$')
yTickMarks = datasets; 
ax.set_yticks(ind+width)
ytickNames = ax.set_yticklabels(yTickMarks)
plt.setp(ytickNames, rotation=45, fontsize=10)

## add a legend
ax.legend( (DCT[0], KNN[0], RBFSVM[0], LINEARSVM[0], POLYSVM[0],GAUSS_NB[0],BERNOUL_NB[0], NDM[0]), labels ) #MLP[0]

plt.show()
