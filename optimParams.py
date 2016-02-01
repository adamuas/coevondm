import datasets;
import dictWriter;
import csvReader;
import preprocessing;
import datasets;
import artificial_dataset;

#Note: Make a copy of this file for each experiment with the appropriate settings.

##soybean dataset
#num_params_in = 35;
##soybean_train = csvReader.read_csv_dataset_right('../../datasets/soybean/soybean_train_small.csv',',',num_params_in);
##soybean_test = csvReader.read_csv_dataset_right('../../datasets/soybean/soybean_test.csv',',',num_params_in);
#
#
##glass dataset
## inputs, 7 outputs
##num_params_in = 9;
#glass_all = csvReader.read_csv_dataset_right('datasets/glass/glass_all.csv',',',num_params_in);
#
##Diabetes dataset
##8 inputs, 1 output
#num_params_out = 8;
#diabetes_train = csvReader.read_csv_dataset_right('datasets/diabetes/diabetes_train.csv',',',num_params_out);
#diabetes_all = csvReader.read_csv_dataset_right('datasets/diabetes/diabetes_all.csv',',',num_params_out);
#
##hepatitis dataset
## 1 output
#num_params_out = 1;
#hepatitis_all = csvReader.read_csv_dataset('datasets/hepatitis/hepatitis_all.csv',',',num_params_out);
#
##abalone dataset
## ?inputs, 1 output
#num_params_out = 1;
#abalone_train = csvReader.read_csv_dataset('datasets/abalone/abalone_train.csv',',',num_params_out);
#abalone_test = csvReader.read_csv_dataset('datasets/abalone/abalone_test.csv',',',num_params_out);
#
##australian credit card dataset
##19 inputs, 1 output
#num_params_in = 14;
#aussie_cc_train = csvReader.read_csv_dataset_right('datasets/australian-credit-data/australian_train.csv',',',num_params_in);
#aussie_cc_test = csvReader.read_csv_dataset_right('datasets/australian-credit-data/australian_test.csv',',',num_params_in);
#aussie_cc_all = csvReader.read_csv_dataset_right('datasets/australian-credit-data/australian_all.csv',',',num_params_in);
#
##sonar dataset
#num_params_in = 60;
#sonar_train = csvReader.read_csv_dataset_right('datasets/sonar/sonar_train.csv',',',num_params_in);
#sonar_test = csvReader.read_csv_dataset_right('datasets/sonar/sonar_test.csv',',',num_params_in);
#sonar_all = csvReader.read_csv_dataset_right('datasets/sonar/sonar_all.csv',',',num_params_in);
#
# #iris dataset
# num_params_in = 4;
# iris_train = csvReader.read_csv_dataset_right('datasets/iris/iris1.csv',',',num_params_in);
# iris_test = csvReader.read_csv_dataset_right('datasets/iris/iris2.csv',',',num_params_in);
# iris_all = csvReader.read_csv_dataset_right('datasets/iris/iris3.csv',',',num_params_in);


#Artificial Generated Pattern Settings

#N-Bit Parity Problem
Bits = 5;
NoSamples = 105;
parity_train = artificial_dataset.Parity(Bits,NoSamples);
NoSamples = 45;
partity_test = artificial_dataset.Parity(Bits,NoSamples);

#Bit Majority Problem
majority_train = artificial_dataset.Majority(Bits,NoSamples);
majority_test = artificial_dataset.Majority(Bits, NoSamples);


#Girosi function
NoSamples = 20;
girosi_train = artificial_dataset.girosiFunction(NoSamples);
NoSamples = 10000;
girosi_test = artificial_dataset.girosiFunction(NoSamples);

#Gabor function
NoSamples = 20;
gabor_train = artificial_dataset.gaborFunction(NoSamples);
NoSamples = 10000;
gabor_test = artificial_dataset.gaborFunction(NoSamples);

#Sugeno function
NoSamples = 20;
start = 1.0;
end = 6.0;
sugeno_train = artificial_dataset.sugenoFunction(NoSamples,start,end);
NoSamples = 20;
start = 1.0;
end = 6.0;
sugeno_test = artificial_dataset.sugenoFunction(NoSamples,start,end);

#USE GPU
do_gpuDiffEvolve = False; #selects and does differential evolution on several solutions at once on GPU
do_gpuMutation = False; #gpu mutation, selects and mutates several solutions at once
gpu_min_sols = 30; #minimum solutions to run a gpu operation
#USE CPU
do_cpuDiffEvolve = True; #True, to also do differential evolution on CPU, False otherwise
do_cpuMutation = False; #True to do mutation operation on CPU, False otherwise
#Parameter Settings
""" Datasets"""
trainset = datasets.XOR; #dataset for training
testset = datasets.XOR;#dataset for testing

""" Optimisation """
update_gap = 3; #rate of taking readings for graph
max_iter = 50; #maximum iterations
param_min = -0.9; #min boundary for parameters
param_max = 0.9; #max boundary for parameters
#DE PARAMS0`    
alpha = 0.2; # weightig factor of differential evolution
de_selection = 1; #selection method of solution for DE
deIter = 3; #maximum differential evolution generations;
#PSO PARAMS
psoIter = 3; #maximum particle swarm optimisation iterations

""" Cross-Over Parameters"""
prob_cross_over = 0.2;  # probability of cross over

"""Mutation Parameters"""
prob_mutation = 0.2 # mutation probability
gaus_mut_mean = 0.0; #mean of gaussian distrubution - sampled for mutation
gaus_mut_std  = 0.2; #standard deviation of gauss distrb. - sampled for mutation
mutation_range = 0.2; #the range of the mutation on a gene

""" Stop Criteria """
target_err = 0.0; # target error
target_errE = 0.1; #target error of ensemble
target_err_test = 0.1;
""" Population """
pop_size = 40; #population size
percent_trim = 0.3; #percentage to eliminate
prob_elim = 0.9; #probability of eliminate
min_age_elim = 3;#minimum age of solutions to eliminate
min_cost = 0.66; #minimum cost of solutions to eliminate
next_genSize = 20; #number of next generation solutions

""" META """
verbose = True; # progress updates
debug = False; # True - prints variable values as the program runs

""" Evolutionary operators """
do_diffEvolution = True;
do_pso = False;
do_crossOver = True;
do_mutation = True;
do_gauss_mut = True;
do_norm_mut = True;

"""" ENSEMBLE PARAMETERS """
use_ensembles = False; #use ensembles
ensemble_size = 20; #Maximum size of the ensemble
use_specialization = False; #use a method of training where networks are specialise to their area of inclination
#Combination methods
#1 - Averaging
#2 - Max
#3 - Min
#4 - Product Rule
combination_method = [1]; #combination methods to be used
#Selection methods
# 1 - Top N solutions
# 2 - Non Dominated Solutions
# 3 - Greedy-Search Selection
selection_method = 1;
#training odd even training
use_odd_even = False;
do_ensemble_mutate = True;
#coefficients of the errors
gamma = 1.0; # coefficient of problem error
beta = 1.0;  # coefficients of ensemble diversity
""" Complimentarity """
theta = 0.2;
min_compliment = 0.5;
minOverlapPerPattern = 0.1;










