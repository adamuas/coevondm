"""
@Description: An implementation of Differential Evolution in python to run on a GPU device.
@Author: Abdullahi Adam
@Pre-requisite Modules: pytools, pycuda, numpy and matplotlib
"""

import os, sys, inspect,traceback;
sys.path.insert(0,"../"); #use default settings
sys.path.insert(0,"../core");
sys.path.insert(0,"../visualisation");
sys.path.insert(0,"../datasets");
sys.path.insert(0,"../cuda");
sys.path.insert(0,"../tools");


#lib imports (includes third party)
import copy;
import time;
import logging;
from random import shuffle;
logging.basicConfig(filename='logger.log',level=logging.DEBUG);


#genome and network
from optimEnDec import *;
from ndm import ndm;
import netParams;
import constants;


#gpu stuff
import cudaInterface;
import pycuda.gpuarray as gpuarray;
import pycuda.driver as cuda;
import pycuda.autoinit;
import numpy;
from pycuda.compiler import SourceModule;
#multiprocessing
import multiprocessing;
import threading;

#complimentary classification
from complimentarity import *;

#plot stuff
import matplotlib.pyplot as plt;
import matplotlib.animation as anim;
import pylab;
from pylab import *;
import arch_error_graph;




#from pyqtgraph.Qt import QtGui, QtCore
#import numpy as np
#import pyqtgraph as pg

#import settings file
import optimParams;


class gpuOptim:
    

    def __init__(self):
        
        #plotting
        self.fig = plt.figure('Error Graph');
        self.best_err_line = None;
        self.worst_err_line = None;
        self.update_gap = optimParams.update_gap;
        
        #dataset
        self.trainset = optimParams.trainset; #dataset to be used for training
        self.testset = optimParams.testset; #data set to be used for testing
        self.lock = threading.Lock();
        
        #ensembles
        self.use_ensembles = optimParams.use_ensembles; #enable ensembles
        self.use_odd_even = optimParams.use_odd_even; #use odd and even training
        self.combination_method = optimParams.combination_method; #combination methods either product,min,max, averaging...
        self.selection_method = optimParams.selection_method; #selection method
        self.best_indErr = 1.0; #best error within the population of the ensembles
        self.best_indErrLst = []; #history of the best errors
        self.curr_spread = 0.0;
        self.ensemble_error = 1.0; #current ensemble error
        self.ensemble_errorLst = []; #list of error through the optimization
        self.ensembles_size = optimParams.ensemble_size; #ensemble size
        self.ensembles_outVar = 0.0; #average output disagreement of the ensembles over the given pattern.
        self.ensembles_outVarHist = []; #history of output disagreements
        self.ensemble_net = []; #selected solutions for the ensemble
        self.ensemble_sols = []; #raw genes of the members of the ensmeble
        self.gamma = optimParams.gamma;
        self.beta = optimParams.beta;
        self.min_compliment = optimParams.min_compliment; #level of complimentarity require (minimum) - ranges between 0.0 - 1.0, where 0.0 - no complimentarity, 1.0 full complimentarity
        self.theta = optimParams.theta; # acceptable distance of outputs on two networks on a pattern to be classified as complimentary
        self.do_ensemble_mutate = optimParams.do_ensemble_mutate;
        
        
        #GPU
        self.do_gpuDiffEvolve = optimParams.do_gpuDiffEvolve;
        self.do_gpuMutation = optimParams.do_gpuMutation;
        self.gpu_min_sols = optimParams.gpu_min_sols;
        
        #CPU
        self.do_cpuDiffEvolve = optimParams.do_cpuDiffEvolve;
        self.do_cpuMutation = optimParams.do_cpuMutation;
        
        #Network- cardinal
        self.useWeightFns = [];
        self.useNodeFns = [];
        
        
        #Optimisation settings
        self.iter = optimParams.max_iter ; #maximum iterations
        self.deIteration = optimParams.deIter; #max differential evolution iterations
        self.pso = optimParams.psoIter; #max iterations for particle swarm optim
        self.target_error = optimParams.target_err; # targer error
        self.pop_size = optimParams.pop_size; #size of population
        self.next_genSize = optimParams.next_genSize; #size of next generation
        self.prob_cross_over = optimParams.prob_cross_over; #probability of crossover
        self.prob_mutation = optimParams.prob_mutation; #probablity of mutation
        self.verbose = False ;#optimParams.verbose; #status of optimisation, show?
        self.debug = False; #print variable values
        self.alpha = optimParams.alpha; #significant of velocities for differential evolution
        self.param_min = optimParams.param_min; #parameter min
        self.param_max = optimParams.param_max; #parameter max
        self.do_diffEvolution = optimParams.do_diffEvolution; #does differential evolution
        self.do_crossOver = optimParams.do_crossOver; #cross over 
        self.do_mutation = optimParams.do_mutation; #mutation
        self.gaus_mut_mean = optimParams.gaus_mut_mean; #mutaion mean
        self.gaus_mut_std = optimParams.gaus_mut_std; #mutation std. deviation
        self.mutation_range = optimParams.mutation_range; #mutation range
        self.percent_trim = optimParams.percent_trim; #percentage of solutions to eliminate
        self.prob_elim = optimParams.prob_elim; #probability of elimination
        self.vector_len = 90; #parameter length
        self.diffBySimilarity = False; #differential evolution selection based on similarity
        self.crOBySimilarity = False; #cross over selection by similartiy
        self.min_age_elim = optimParams.min_age_elim; #minimum age of solutions to be eliminated
        self.min_cost_elim =  optimParams.min_cost; #minimum cost of solutions to be evaluated
        self.use_cardinal = netParams.use_cardinal;
        
        #cache variables
        self.sols = []; # is a list of the solutions
        self.gpuSols = array([]); # copy of solutions list on gpu
        self.worst_cost = 1.0;  #globally worst solution
        self.best_cost = float('inf'); #globally best cost
        self.best_cost_test = 1.0; #best test error
        self.worst_cost_test = 1.0; #worst error on test set
        self.worst_sol_index = ''; #worst solutions index
        self.best_sol= []; #best solution on training set
        self.best_sol_test = []; #best solution on test set
        self.worst_sol = []; #worst solution
        self.lst_best_cost = list(); #list of best cost
        self.lst_best_cost_test = list(); #list of best cost
        self.lst_worst_cost = list(); #list of worst cost
        self.generations = list();
        self.curr_iter = 0;
        #genetic sensitivy mask
        self.geneSensitivityMask = []; # sensitivity of output to each gene
        
        
        

    def optimise(self):
        """ optimise the solutions """

        
        current_iter = 0;
        endSearch = False;
        
        #graph
        if self.use_ensembles:
            #Error (Graph)
            bc = self.fig.add_subplot(121);
            ion();
            plt.title('Best Cost(MSE)');
            plt.show(block=False);
            plt.ylabel('Error(MSE)');
            plt.xlabel('Generations');
            plt.grid(True);
                
            #Diversity graph
            ed = self.fig.add_subplot(122);
            ion();      
            plt.title('Population train(f1) and treated train(f2) Errors ');
            plt.show(block=False);
            plt.xlabel('f1');
            plt.ylabel('f2');
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.grid(True);
        else:
            bc = self.fig.add_subplot(111);
            ion();
            plt.title('Best Cost(MSE)');
            plt.show(block=False);
            plt.ylabel('Error(MSE)');
            plt.xlabel('Generations');
            plt.grid(True);
        
        
        
        
        #use cardinal of transfer functions
        if self.use_cardinal:
            #get weight functions
            weightFns = list(netParams.weightFns);
            nodeFns = list(netParams.nodeFns);
            print ">>>NOTE: PLEASE CHECK AND MAKE SURE IF YOU WANT TO USE USE_CARDINAL";
            if self.debug:
                print ("Designated set of Weight and Node functions:");
                print weightFns;
                print nodeFns;
            
            #get cardinal
            numWFns = netParams.weightFnsToUse;
            numNFns = netParams.nodeFnsToUse;
            
            #shuffle
            shuffle(weightFns);
            shuffle(nodeFns);
            
            #pick cardinal
            weightFns = weightFns[:numWFns];
            nodeFns = nodeFns[:numNFns];
            
            #set for optim
            self.useNodeFns = nodeFns;
            self.useWeightFns = weightFns;
            
            if self.debug or self.verbose:
                print (">>cardinal:");
                print ("-weight functions:", self.useWeightFns);
                print ("-node functions:", self.useNodeFns);
            
        """ Population Initialisation """
        self.pop_init();
       
        """ OPTIMISATION"""
        while not endSearch:
            
            
            """ Evaluation """
            #evaluate the solutions
            #self.evaluate(self.trainset);
            
            print "-EVALUATION"
            self.evaluate_multiprocessor(self.trainset);
            
            
            #f1,f2 = self.getObjcosts();
            #print f1;
            #print f2;
            #if current_iter == 10:
            #    break;
            #
            
            
            #validate the solutions
            #self.validate(self.trainset);
            
            """ Sorting"""
            #Sort solutions
            self.sortSolutions();
            
            
            """ DIFFERENTIAL EVOLUTION"""
            print ">>> DIFFERENTIAL EVOLUTION"
            #DIFFERENTIAL-EVO:perform differential evolution of solutions
            
            if self.do_diffEvolution:
                #GPU Differential Evolutions
                if self.do_gpuDiffEvolve and len(self.sols) > self.gpu_min_sols:
                    gpu_diffSols = self.gpu_diffEvolve();
                    if gpu_diffSols != None:
                        print ("--used gpu diff evolution");
                        self.sols.extend(gpu_diffSols);
                
                #CPU Differential Evolution
                if self.do_cpuDiffEvolve:
                    diffSols = self.diffEvolve();
                    if diffSols != None:
                        print ("--used cpu diff evolution");
                        self.sols.extend(diffSols);

            """ PARTICLE SWARM OPTIMISATION """
            #perform particle swarm optimisation on solutions
            #self.particleSwarm();
            
            """ CROSS OVER"""
            #CROSS-OVER : perform cross over operation
            print ">>>CROSS-OVER"
            if(self.do_crossOver == True):
                coSol = self.crossOver();
                if coSol != None:
                    self.sols.append(coSol);

    
            """ MUTATION """
            #MUTATION: perform mutation over solutions
            if(self.do_mutation == True):
                
                #GPU mutation
                if self.do_gpuMutation and len(self.sols) > self.gpu_min_sols :
                    gpu_mut_sols = self.gpu_mutate();
                    if gpu_mut_sols != None:
                        print ("--used gpu mutation");
                        self.sols.extend(gpu_mut_sols);
                    
                #CPU mutation 
                if self.do_cpuMutation:
                    #CPU Mutation(Gaussian)
                    mut_sol = self.gauss_mutate();
                    if mut_sol != None:
                        print ("--used cpu mutation(gauss)");
                        self.sols.append(mut_sol);
                    
                    # CPU mutation (Random within range)
                    mut_sol = self.mutate();
                    if mut_sol != None:
                       print "-used cpu mutation()";
                       self.sols.append(mut_sol);
                    

            """ ENSEMBLES"""
            #-evaluate
            if self.use_ensembles == True and current_iter % 30 == 0:
                #select members of the ensemble
                if  self.selection_method == constants.GREEDY_SELECT: #3
                    self.ensemble_member_select2();
                elif self.selection_method == constants.NON_DOMINATED_FRONT: #2
                    self.non_dominated_select();
                elif self.selection_method == constants.TOP_N: #1
                    self.sortSolutionsByTestError();
                    self.ensemble_sols =self.sols[:self.ensembles_size];
                    
               
                #evaluate ensemble
                self.ensemble_evaluate(self.trainset);
                

            """ VISUALISATION AND OTHER HEAVY FUNCTIONS"""
            if current_iter % self.update_gap == 0:
                
                
                #generations
                self.generations.append(current_iter);
                #get objective costs for pareto vis
                f1,f2 = self.getObjcosts();
                
                
                #add best cost at generation
                self.lst_best_cost.append(self.best_cost);
                #add best cost (test) error at generation
                self.lst_best_cost_test.append(self.best_cost_test);
                #add worst cost at generation
                self.lst_worst_cost.append(self.worst_cost);
             
                
                #Ensembles
                if self.use_ensembles:
                    #add ensembler error
                    self.ensemble_errorLst.append(self.ensemble_error);
                    #add disagreement of ensembles
                    self.ensembles_outVarHist.append(self.ensembles_outVar);
                    #add the error of the best indiviudual within the ensemble
                    self.best_indErrLst.append(self.best_indErr);
                    
                if self.debug or True:
                    print ("*Best Cost:", self.best_cost);
                    
                
                #update graph
                #-draw best cost graph
                bc.errorbar(self.generations, self.lst_best_cost,fmt='--o',color = 'b', label = 'Best Cost - Train');
                bc.errorbar(self.generations,self.lst_best_cost_test,fmt='-o',color='y',label = 'Best Cost - Test');
                if self.use_ensembles:
                    bc.errorbar(self.generations, self.ensemble_errorLst,fmt='--o', color ='r',label='Ensemble Error-Test'); #Ensemble error (test set)
                    bc.errorbar(self.generations,self.best_indErrLst,fmt='-o',color = 'r', label ='Best Individual-Test'); # Error of best individual in the ensemble
                    # ERROR DISAGREEMENT
                    #ed.errorbar(self.generations,self.ensembles_outVarHist, fmt='--o',color = 'y');
                    ed.scatter(f1,f2);
                
                ##get solution architectures, errors and size of architecture colonies
                #sols_arch,errors,arch_colony_sizes = arch_error_graph.get_archs_errors(self.sols);
               
                # size in points ^2
                #arch_error_graph.norm_size(arch_colony_sizes);
                #magnify = 30;
                #col_size = (magnify* arch_colony_sizes);
                #gen = numpy.ones((len(errors))) * current_iter;
    
                ##draw error distribution graph
                #ed.scatter(gen, errors,c=sols_arch,s=col_size,  alpha=0.75);
                
                ##label all of the scatter points
                #if current_iter <= 1:
                #    offset = -2.5;
                #    for i in xrange(len(sols_arch)):
                #        ed.text(gen[i]+offset, errors[i], '%d'%int(sols_arch[i]),
                #            ha='center', va='top',color='brown', fontsize=10);
                #update graphs
                plt.draw();

            
            """ SURVIVAL OF FITTEST """
            #eliminate some solutions
            print ("SOLS BEFORE:", len(self.sols));
            self.trimSolutions();
            print ("SOLS AFTER:", len(self.sols));
            #age solutions
            self.ageSolutions();
            
            """ CREATE NEW GENERATION OF SOLUTIONS """
            #create new solutions
            nwSols = self.createSols(optimParams.next_genSize);
            self.sols.extend(nwSols);
            
            
            #increament the iterations/generation                
            current_iter += 1;
            self.curr_iter = current_iter;            
            
            #end search if the max iteration limit is met
            if current_iter >= self.iter:
                endSearch = True;
                
            #end search if the target best error is met
            if self.use_ensembles == False and self.best_cost <= self.target_error and self.best_cost != -1:
                endSearch = True;
     
            #end search if the target error of ensemble is met
            if self.use_ensembles == True and self.best_cost <= self.target_error and self.ensemble_error == optimParams.target_errE \
            and self.best_cost != -1 :
                endSearch  = True;
                
        print (">>Optimisation Complete!!!");
        
        """Save the Graph """
        timestamp = getStrTimeStamp();
        fig_name = 'optimGraph_' + timestamp +'.jpg';
        plt.savefig(fig_name);
        
        """Population evaluation """
        self.evaluate(self.testset);
        self.best_cost,b_soli,self.best_sol = self.getBest();
        self.worst_cost,self.worst_sol_index,self.worst_sol = self.getWorst();
        self.best_cost_test,self.worst_cost_test = self.validate(self.testset,All=True);
        
        """ Ensemble evaluation"""
        if self.use_ensembles :
            #select members of the ensemble
            if  self.selection_method == constants.GREEDY_SELECT:
                self.ensemble_member_select();
            elif self.selection_method == constants.NON_DOMINATED_FRONT:
                self.non_dominated_select();
            elif self.selection_method == constants.TOP_N:
                self.sortSolutionsByTestError();
                self.ensemble_sols =self.sols[:self.ensembles_size];
            #evaluate ensemble
            self.ensemble_error,self.ensembles_outVar = self.ensemble_evaluate(self.testset);
        
        """Final Results """
        print ("### FINAL RESULTS ###");
        print ("-Population");
        print ("Best Cost(Train Set):", self.best_cost);
        print ("Best Cost(Test Set):", self.best_cost_test);
        print ("Worst Cost(Train Set):", self.worst_cost);
        print ("Worst Cost(Test Set):", self.worst_cost_test);
        
        if self.use_ensembles:
            print ("-Ensemble");
            print ("Ensemble Error(Test set):", self.ensemble_error);
            print ("Best Ensemble Member(Test Error):", self.best_indErr);
            print ("Ensemble Size:", len(self.ensemble_sols));
        
     
      
        
    def updateErr(self,errs):
        """ graph update """
        self.err_line.set_ydata(errs);
        
    def getErr(self):
        return self.best_cost;

    def pop_init(self, vec_length = None):
        """ intialise the number of solutions allowed  with a given length """
        if vec_length == None :
            vec_length = self.vector_len;

        #create solutions
        if(len(self.sols) == 0):
            self.sols = self.createSols(optimParams.pop_size);
        else:
            pass;
            #self.sols = numpy.random.randn(vec_length, self.pop_size).astype(numpy.float32);

        #verbose
        if self.verbose == True:
            print "\n -Population intialised.";
            
    def createSols(self, num_sols,m=None,v=None):
        """ creates and return new soluitons based on specification of the netParams file """
        #create sols
        sols = [];
        
        #create solutions
        for i in xrange(num_sols):
            #generate genes
            if self.use_cardinal:
                genes = generateGenes(self.useNodeFns,self.useWeightFns,False,True);
            else:
                genes = generateGenes();
            
            #append new solution
            sols.append(genes);
            
        if self.verbose == True:
            print ("\n -sols created");
            
            
        return sols;
            

    def setPopulation(self, lst_sols):
        """ gets the list of genomes"""
        self.sols = lst_sols;

    def addSol(self, sol):
        """ gets the list of genomes"""
        self.sols.append(sol);

    def particleSwarm(self, pop = None):
        """ evolve using particle swarm optimisation"""
        if(pop == None):
            pop = self.sols;
        
        #number of solutions
        num_sols = len(pop);
        
        #number of solutions required
        num_req_sols = 3;
        
        #solutions list
        nwSols = [];
      
        #copy solutions to gpu
        try:

            #check if there are solutions
            if num_sols < 0 :
                raise "\n -Error: No Solutions.";

            if self.verbose == True:
                print ("\n -Starting Differential Evolution");
        
            for i in xrange(self.psoIter):

                
                #selection of solutions               
                if(self.diffBySimilarity == True):
                     #Niched Differential Evoution alogrithm
                    pass;
                else:
                    # Randomly
                    selectedSols,selectedSolInd = self.randomSelection(num_req_sols,pop);
                
                #get solutions though without cost and fitness
                solX = self.sols[x][2:];
                solY = self.sols[y][2:];
                solZ = self.sols[z][2:];

                if self.debug == True:
                    print ("\n -SolX:", solX);
                    print ("\n -SolY:", solY);
                    print ("\n -SolZ:", solZ);

                #to-do, establish leader and followers in solution.
                solsIndx = [x, y, z];
                costs = [solX[constants.COST_GENE], solY[constants.COST_GENE], solZ[constants.COST_GENE]];
                
                leaderIndx  = costs.index(min(costs));
                otherIndx  = costs.index(max(costs));
                
                
                #copy to gpu
                if self.verbose == True:
                    print ("\n -Copying to GPU");

                #copy solution to gpu
                gpuSolX = gpuarray.to_gpu(solX);
                gpuSolY = gpuarray.to_gpu(solY);
                gpuSolZ = gpuarray.to_gpu(solZ);

                #calculate mutate solution
                #TODO - make sure to normalise the values after this operation.
                mutantSol = (gpuSolX + self.alpha * (gpuSolY - gpuSolZ)).get();
                

                #Set mutant cost
                mutantSol[0] = -1;

                #TODO - grand velocity and generate another offpring
                
                #add solution to the end of the solution.
                #pop.resize(num_sols+1 ,vec_len);
                #pop[-1] = mutantSol;

                
                
                if self.debug == True:
                    print ("\n -SolX:", solX);
                    print ("\n -SolY:", solY);
                    print ("\n -SolZ:", solZ);
                    print ("\n -Mutant Solution:", mutantSol);

                return mutantSol;
  
        
        except:
            print ("\n-Error: Could not copy solutions to gpu.");
            print traceback.print_exc();

    def gpu_diffEvolve(self):
        """ gpu diff evolution"""
        
        #for debugging locally
        debug = False;
        
        #minimum solutions required
        min_sols_req = 9;
        num_sol_partitions = 3;
        partition_point = 0;
        
        #solutions
        sol_leaders = [];
        sol_followers = [];
        sol_mutants = [];

        #solutions to be returned
        nwSols = [];
        
        
        #number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H']  + netParams.nodeConfig['O'];
        
        """ Do differential evolution excluding metation information and architectural information """
        DiffEvolveFrom = constants.META_INFO_COUNT + num_nodes;
        
        #select random solution
        one_random_sol = 1;
        sel_sols,sel_soli = self.randomSelection(min_sols_req);
       
        
        #no solutions found
        if sel_sols == None and sel_soli == None:
            return None;
        
        #minimum number of solutions require
        if len(sel_sols) <= min_sols_req:
            return None;
        
        #sort solutions
        self.sortSolutions(sel_sols);
        
        
        #partition selected soltons into leaders, followers and mutants
        #-make it divisible into the partitions
        while len(sel_sols)% num_sol_partitions != 0:
            #dump one solution
            sel_sols.pop();
            
        #select partition point
        partition_point = len(sel_sols)/num_sol_partitions;
        
  
        
        #leaders of solutions
        sol_leaders = numpy.array(sel_sols[:partition_point]);
        #follower solutions
        sol_followers = numpy.array(sel_sols[partition_point:(partition_point*2)]);
        #solutions to mutate
        sol_mutants = numpy.array(sel_sols[(partition_point*2):(partition_point*3)]);
        
        if debug == True:
            print ("leaders:",sol_leaders);
            print ("followers:",sol_followers);
            print ("mutants:",sol_mutants);
        
        
        #do differential evolution on GPU    
        gpu_diffSols = cudaInterface.cuda_diffEvo(sol_leaders,sol_followers,sol_mutants,self.alpha);
        
        #meta genotypes
        cost = -1.0;
        age = 0.0;
        misc = 0.0;
        
        for i,sol in enumerate(gpu_diffSols):
            sol[constants.COST_GENE] = cost;
            sol[constants.COST2_GENE] = cost;
            sol[constants.MISC_GENE] = misc;
            sol[constants.AGE_GENE] = age;
            
        
        if  debug == True:
            print ("\ngpu_sols:", gpu_diffSols);
        
        
        #return new solutions
        return gpu_diffSols;
        
        
    def diffEvolve(self,pop = None):
        """ evolve using differential evolution """
        if(pop == None):
            pop = self.sols;
            
        ##logging
        #logging.info("-Differential Evolution");

        #get the populations dimensions
        num_sols = len(pop);

        #number of required solutions
        num_req_sols = 3;
        
        #new solutions list
        nwSols = [];
      
        #number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H']  + netParams.nodeConfig['O'];

        #grand velocity stuff
        velocities = [];
        
        """ Do differential evolution excluding metation information and architectural information """
        DiffEvolveFrom = constants.META_INFO_COUNT + num_nodes;
       
        try:

            #check if there are solutions
            if num_sols < 0 :
                raise "\n -Error: No Solutions.";

            if self.verbose == True:
                print ("\n -Starting Differential Evolution");
        
            for i in xrange(self.deIteration):

                #selection of solutions               
                if(self.diffBySimilarity == True):
                     #Niched Differential Evolution alogrithm
                    pass;
                else:
                    # Randomly
                    selectedSols,selectedSolInd = self.randomSelection(num_req_sols,pop);
                
                #if there are no solutions of same length found for DE skip DE run for the time
                if (selectedSols == None and selectedSolInd == None):
                    #set no solutions
                    nwSols  = None;
                    break;
                     

                ##sort according to fitness
                selectedSols = self.sortSolutions(selectedSols);

                followerSol = selectedSols[1][DiffEvolveFrom:]; #2nd follower
                leaderSol = selectedSols[0][DiffEvolveFrom:]; #leader
                followerSol2 = selectedSols[2][DiffEvolveFrom:]; #1st follower
                
                if self.debug == True:
                    print "solLeader",leaderSol;
                    print "solFollower",followerSol;
                    print "solFollower2",followerSol2;
                    
                    #log their lengths and architecture
                    print ('LeaderSol len: %d', len(selectedSols[0]));
                    print('LeaderSol Arch.: %d', (selectedSols[0][constants.META_INFO_COUNT:DiffEvolveFrom ]));
                    print('follower len: %d', len(selectedSols[1]));
                    print('follower Arch.: %d', (selectedSols[1][constants.META_INFO_COUNT:DiffEvolveFrom ]));
                    print('follower2 len: %d', len(selectedSols[2]));
                    print('follower2 Arch.: %d', (selectedSols[2][constants.META_INFO_COUNT:DiffEvolveFrom ]));
                
                #calculate mutate solution
                #TODO - make sure to normalise the values after this operation.
                velocity = self.alpha * (array(leaderSol) - array(followerSol));
                
                mutantSol = array(followerSol2) + self.alpha * (array(leaderSol) - array(followerSol));
                mutantSol.tolist();


                #META-INFO
                nwSol = [];
                nwSol.insert(constants.COST_GENE,-1.0);
                nwSol.insert(constants.COST2_GENE,-1.0);
                nwSol.insert(constants.MISC_GENE,0.0);
                nwSol.insert(constants.AGE_GENE,0);
                
                
                #Architecture
                nwSol.extend(selectedSols[1][constants.META_INFO_COUNT : DiffEvolveFrom]);
                nwSol.extend(mutantSol);

                if self.debug == True:
                    print "\nmutant sol:", nwSol;
                    print('mutant len: %d', len(nwSol));
                    print('mutant Arch.: %d', (nwSol[constants.META_INFO_COUNT:DiffEvolveFrom ]));
                
                #TODO - grand velocity and generate another offpring
                
              
                #add solution
                nwSols.append(nwSol);
                
                #print result
                if self.debug == True:
                    print "\n -SolX:", solX;
                    print "\n -SolY:", solY;
                    print "\n -SolZ:", solZ;
                    print "\n -Mutant Solution:", mutantSol;


            return nwSols;

        except:
            #logging.error('error while runing DE');
            print traceback.print_exc();
        
    
                
    """ ENSEMBLE METHODS """
    def combineOutputs(self,lstOutputs,fitness=None):
        """ combines the outputs of solutions"""
        
        #outputs
        outs = dict();
        outs['outAvg']= 0.0;
        outs['outMax'] = 0.0;
        outs['outMin'] = 0.0;
        outs['outProd'] = 0.0;
        
        #calculate contribution power by fitness
        if fitness != None:
            contrb_pow = numpy.zeros(len(lstOutputs));
            cumm_sum = sum(fitness);
            for i,f in enumerate(fitness):
                contrb_pow[i] = (f/cumm_sum);
            
            #debug or verbose
            if self.debug or self.verbose:
                print "contribution powers", contrb_pow;
                
            #multiply outputs by contribution powers
            lstOutputs = list(lstOutputs * contrb_pow);
            
        #Averaging Rule
        if 1 in self.combination_method :
            outs['outAvg']= sum(lstOutputs )/float(len(lstOutputs));
        #Max Rule
        if 2 in self.combination_method :
            outs['outMax'] = max(lstOutputs );
        #Min Rule
        if 3 in self.combination_method :
            outs['outMin'] = min(lstOutputs);
        #Product Rule
        if 4 in self.combination_method :
            outs['outProd'] = 1.0;
            for x in lstOutputs:
                outs['outProd']  *= x;
        
        return outs;
    
    def non_dominated_select(self, pop=None):
        """ selects the non-dominated members of the population as the ensemble members """
        
        if pop == None:
            pop = self.sols[:];
        
        
        Sp = [];
        F = [];
        np = 0;
        nq = 0;
        
        #search for the non-dominated solutions
        for soli,solx in enumerate(pop):
            
            
            p1 = solx[constants.COST_GENE];
            p2 = solx[constants.COST2_GENE];
            
            if p1 == constants.NOT_EVALUATED or p2 == constants.NOT_EVALUATED:
                continue;
            
            for solj,soly in enumerate(pop):
                
                if soli == solj:
                    continue;

                q1 = soly[constants.COST_GENE];
                q2 = soly[constants.COST2_GENE];
                
                if q1 == constants.NOT_EVALUATED or q2 == constants.NOT_EVALUATED:
                    continue;
                
                
                if p1 < q1 and p2 < q2 : #p dominates q
                    #add to dominated
                    Sp.append(solx);

                elif q1 < p1 and q2 < p2: #q dominates p
                    np +=1;
                    
            if np == 0:
                F.append(solx);
                

        print ">>Front :", F;
        #copy to ensemble
        self.ensemble_sols = list(F);    
        
    def ensemble_member_select(self, pop= None):
        """ selects solutions of the members of the ensemble """
        max_tries = 3;
        use_sol = False;
        minErr = 0.3;
        
        if self.verbose or self.debug or True:
            print ">>Ensemble member select";
                
        if len(self.ensemble_sols) == 0:
            """select the first member of the ensemble from the population that minimizes ensemble error
            """
    
            #sort the solutions
            self.sortSolutions();
            
            while True:
                #try another random solution
                cand_soli = numpy.random.rand() * (len(self.sols)-1);
                cand_sol = self.sols[int(cand_soli)];
                cand_sol_cost = cand_sol[constants.COST2_GENE]; #test error
                #check if the solution is not in the ensemble and if it is not just a random guesser
            
                if cand_sol not in self.ensemble_sols and cand_sol_cost < minErr :
                    use_sol = True;
                    break;
                
                #infinite loop control
                max_tries -= 1;
                if max_tries == 0:
                    break;
                
            #copy solution
            if use_sol == True :
                nwmember = self.sols[int(cand_soli)][:];
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(nwmember,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(nwmember);
                    
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #EVALUATE TO CHECK FOR INCREASE IN DIVERSITY AND DECREASE IN ERROR
                 #evaluate on the test set
                errCand = net.evaluate(self.testset,None,False,True);

                
                if self.debug or self.verbose or True:
                    print ">>Candidate Solution."
                    print "Ind. Error(Test)",errCand;
                    print "Ind. Error(Train)",cand_sol_cost;
                
                if errCand < minErr :
                    #keep the solution in the ensemble
                    self.ensemble_sols.append(nwmember);
                    if True or self.verbose or self.debug == True:
                        print ">>Ensemble Error and Diversity Increased.";
                else:
                    if True or self.verbose or self.debug == True:
                        print ">>Ensemble Error and Diversity Diminished.";
                        
                        
        elif len(self.ensemble_sols) < self.ensembles_size:
            """ find complimentarity solutions to the first solution """
            
            is_complimentary = False;
            MAX_TRIES = 7;
            
            subject_soli = int(numpy.random.rand() * len(self.ensemble_sols));            
            subject_sol = self.ensemble_sols[subject_soli];
            
            #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
            #get the genome of the solution
            if self.use_cardinal:
                sol_genome = decodeGenes(subject_sol,self.useNodeFns,self.useWeightFns);
            else:
                sol_genome = decodeGenes(subject_sol);
                
            #create temp net
            net = ndm();
            #recreate network with the genome
            net.recreateNet(sol_genome);
                
            #evaluate on the test set
            errSub,outputPatSub = net.evaluate(self.trainset,None,True,True);
            
            if self.debug or self.verbose:
                print "output pattern", outputPatSub;
            
            #delete
            del net;
            
            #is not complimentary
            while True:
                
                #pick a solution randomly
                comp_soli = int(numpy.random.rand() * len(self.sols));            
                comp_sol = self.sols[comp_soli];
                
                #if its already in the ensemble skip it
                if comp_sol in self.ensemble_sols:
                    print "-Solution already in ensemble, skipping."
                    continue;
                
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(comp_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(comp_sol);
                    
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errComp,outputPatComp = net.evaluate(self.trainset,None,True,True);
                
                #delete
                del net;
                
                #comput complimentarity of the outputs on the test error
                comp_overlap = computeOverlap(outputPatSub,outputPatComp,self.testset['OUT'],self.theta);
                
                
                if self.debug or self.verbose or True:
                    print "comp_overlap(population candidate)", comp_overlap;
                
                
                #decreament max tries
                MAX_TRIES -= 1;
                
                if MAX_TRIES == 0:
                    if self.debug or self.verbose  and is_complimentary or True:
                        print "-No Complimentary Solution found for subject solution";
                    break;
                
                if 1 - comp_overlap >= self.min_compliment and errComp < minErr:
                    is_complimentary = True;
                    #add the solution to the ensemble
                    self.ensemble_sols.append(comp_sol[:]);
                    break;
                    
        elif len(self.ensemble_sols) == self.ensembles_size :
            """ Replace ensemble members with members of the solution that will be able to reduce the
            error of the Ensemble and Increase the Diversity of the Ensemble"""
            
            #select a subject soltuion from the ensemble
            subject_soli = int(numpy.random.rand() * len(self.ensemble_sols));            
            subject_sol = self.ensemble_sols[subject_soli];
            
            #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
            #get the genome of the solution
            if self.use_cardinal:
                sol_genome = decodeGenes(subject_sol,self.useNodeFns,self.useWeightFns);
            else:
                sol_genome = decodeGenes(subject_sol);
                
            #create temp net
            net = ndm();
            #recreate network with the genome
            net.recreateNet(sol_genome);
                
            #evaluate on the test set
            errSub,outputPatSub = net.evaluate(self.trainset,None,True,True);
            
            #del
            del net;
        
            #pick a solution to remove
            to_remove = int(numpy.random.rand() * len(self.ensemble_sols)); 
            while to_remove != subject_soli:
                to_remove = int(numpy.random.rand() * len(self.ensemble_sols));            
            
            to_remove_sol = self.ensemble_sols[to_remove];
            
            #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
            #get the genome of the solution
            if self.use_cardinal:
                sol_genome = decodeGenes(to_remove_sol,self.useNodeFns,self.useWeightFns);
            else:
                sol_genome = decodeGenes(to_remove_sol);
                
            #create temp net
            net = ndm();
            #recreate network with the genome
            net.recreateNet(sol_genome);
                
            #evaluate on the test set
            errToRem,outputPatToRem = net.evaluate(self.trainset,None,True,True);
            
            #del
            del net;
            
            #comput complimentarity of the outputs on the test error
            #comp_degreeRem = computeComplimentarity(outputPatSub,outputPatToRem,self.theta);
            comp_overlapRem = computeOverlap(outputPatSub,outputPatToRem,self.testset['OUT'],self.theta);
            
            #mutate member of the solution to get another possible member
            if self.do_ensemble_mutate :
                mut_sol = self.ensemble_mutate();
            
            #perform cross over of member of the solution to get another possible member
            co_sol = None; #self.ensemble_crossover();
            
            #evaluate with mutant sol
            #(1)add mutant sol to ensemble and then evaluate
            if mut_sol != None:
                
                if self.debug or self.verbose or True:
                    print ">>Using Mutant Solution";
                
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(mut_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(mut_sol);
                    
               
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errMut,outputPatMut = net.evaluate(self.trainset,None,True,True);
                
                #delete
                del net;
                
                #comput complimentarity of the outputs on the test error
                #comp_degreeMut = computeComplimentarity(outputPatSub,outputPatMut,self.theta);
                comp_overlapMut = computeOverlap(outputPatSub,outputPatMut,self.testset['OUT'],self.theta);
                
                if self.debug or self.verbose or True:
                    print "comp_degree(mutant candidate)", comp_overlapMut;
                
                
                #add solution if its complimentarity and error are better
                if  1-comp_overlapMut > self.min_compliment and errMut < errToRem:
                    #add mutant 
                    self.ensemble_sols.append(mut_sol);
                    # remove the solution in question
                    self.ensemble_sols.remove(to_remove_sol);
                    
                    if self.debug or self.verbose or True:
                        print ">> solution replaced by mutant solution. "
                    
           
            if mut_sol == None:
        
                #pick a solution as replacement solution from the population(top half)
                replacement_i = int(numpy.random.rand() * len(self.sols));            
                replacement_sol = self.sols[replacement_i];
                
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(replacement_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(replacement_sol);
                    
               
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errCompCand,outputPatCompCand = net.evaluate(self.trainset,None,True,True);
                
                #delete
                del net;
                
                #comput complimentarity of the outputs on the test error
                #comp_degreeCand = computeComplimentarity(outputPatSub,outputPatCompCand,self.theta);
                comp_overlapCand = computeOverlap(outputPatSub,outputPatCompCand,self.testset['OUT'],self.theta);
                
                if self.debug or self.verbose or True:
                    print "comp_degree(population candidate)", comp_overlapCand;
                
                
                #add solution to ensemble
                if  1-comp_overlapCand > self.min_compliment and errCompCand <  errToRem:
                    #add the solution
                    self.ensemble_sols.append(replacement_sol);
                    # remove the solution in question
                    self.ensemble_sols.remove(to_remove_sol);
                    #debug
                    if self.debug or self.verbose or True:
                        print ">>Replacement Picked from population.";
                    
    def ensemble_member_select2(self, pop= None):
        """ selects solutions of the members of the ensemble """
        max_tries = 3;
        use_sol = False;
        minErr = 0.3;
        
        if self.verbose or self.debug or True:
            print ">>Ensemble member select";
                
        if len(self.ensemble_sols) == 0:
            """select the first member of the ensemble from the population that minimizes ensemble error
            """
    
            #sort the solutions
            self.sortSolutions();
            
            while True:
                #try another random solution
                cand_soli = numpy.random.rand() * (len(self.sols)-1);
                cand_sol = self.sols[int(cand_soli)];
                cand_sol_cost = cand_sol[constants.COST2_GENE]; #test error
                #check if the solution is not in the ensemble and if it is not just a random guesser
            
                if cand_sol not in self.ensemble_sols and cand_sol_cost < minErr :
                    use_sol = True;
                    break;
                
                #infinite loop control
                max_tries -= 1;
                if max_tries == 0:
                    break;
                
            #copy solution
            if use_sol == True :
                nwmember = self.sols[int(cand_soli)][:];
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(nwmember,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(nwmember);
                    
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #EVALUATE TO CHECK FOR INCREASE IN DIVERSITY AND DECREASE IN ERROR
                 #evaluate on the test set
                errCand = net.evaluate(self.testset);

                
                if self.debug or self.verbose or True:
                    print ">>Candidate Solution."
                    print "Ind. Error(Test)",errCand;
                    print "Ind. Error(Train)",cand_sol_cost;
                
                if errCand < minErr :
                    #keep the solution in the ensemble
                    self.ensemble_sols.append(nwmember);
                    if True or self.verbose or self.debug == True:
                        print ">>Ensemble Error and Diversity Increased.";
                else:
                    if True or self.verbose or self.debug == True:
                        print ">>Ensemble Error and Diversity Diminished.";
                        
                        
        elif len(self.ensemble_sols) < self.ensembles_size:
            """ find complimentarity solutions to the first solution """
            
            is_complimentary = False;
            MAX_TRIES = 7;
            
            
            #output patterns in ensemble
            EnOutputPat = [];
            
            #get the output pattern of all the solutions
            for subject_soli,subject_sol in enumerate(self.ensemble_sols):
            
                #subject_soli = int(numpy.random.rand() * len(self.ensemble_sols));            
                #subject_sol = self.ensemble_sols[subject_soli];
                
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(subject_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(subject_sol);
                    
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errSub,outputPatSub = net.evaluate(self.testset,None,True);
                #add output pattern to ensemble output patterns
                EnOutputPat.append(outputPatSub);
                
                if self.debug or self.verbose:
                    print "output pattern", outputPatSub;
                
                #delete
                del net;
            
            #is not complimentary
            while True:
                
                #pick a solution randomly
                comp_soli = int(numpy.random.rand() * len(self.sols));            
                comp_sol = self.sols[comp_soli];
                
                #if its already in the ensemble skip it
                if comp_sol in self.ensemble_sols:
                    print "-Solution already in ensemble, skipping."
                    continue;
                
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(comp_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(comp_sol);
                    
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errComp,outputPatComp = net.evaluate(self.testset,None,True);
                
                #delete
                del net;
                
                #comput complimentarity of the outputs on the test error
                spread = CCsolutionSelect(EnOutputPat,outputPatComp,self.testset['OUT'],self.theta);
                
                
                if self.debug or self.verbose or True:
                    print "spread(population candidate)", spread;
                
                
                #decreament max tries
                MAX_TRIES -= 1;
                
                if MAX_TRIES == 0:
                    if self.debug or self.verbose  and is_complimentary or True:
                        print "-No Complimentary Solution found for subject solution";
                    break;
                
                if spread >= self.curr_spread and errComp < minErr:
                    is_complimentary = True;
                    #add the solution to the ensemble
                    self.ensemble_sols.append(comp_sol[:]);
                    break;
                    
        elif len(self.ensemble_sols) == self.ensembles_size :
            """ Replace ensemble members with members of the solution that will be able to reduce the
            error of the Ensemble and Increase the Diversity of the Ensemble"""
            
            #output patterns in ensemble
            EnOutputPat = [];
            
            #get the output pattern of all the solutions
            for subject_soli,subject_sol in enumerate(self.ensemble_sols):
            
                #subject_soli = int(numpy.random.rand() * len(self.ensemble_sols));            
                #subject_sol = self.ensemble_sols[subject_soli];
                
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(subject_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(subject_sol);
                    
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errSub,outputPatSub = net.evaluate(self.testset,None,True);
                #add output pattern to ensemble output patterns
                EnOutputPat.append(outputPatSub);
                
                if self.debug or self.verbose:
                    print "output pattern", outputPatSub;
                
                #delete
                del net;
        
            #pick a solution to remove
            to_remove = int(numpy.random.rand() * len(self.ensemble_sols)); 
            while to_remove != subject_soli:
                to_remove = int(numpy.random.rand() * len(self.ensemble_sols));            
            
            to_remove_sol = self.ensemble_sols[to_remove];
            
            #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
            #get the genome of the solution
            if self.use_cardinal:
                sol_genome = decodeGenes(to_remove_sol,self.useNodeFns,self.useWeightFns);
            else:
                sol_genome = decodeGenes(to_remove_sol);
                
            #create temp net
            net = ndm();
            #recreate network with the genome
            net.recreateNet(sol_genome);
                
            #evaluate on the test set
            errToRem,outputPatToRem = net.evaluate(self.testset,None,True);
            
            #del
            del net;
            
            #comput complimentarity of the outputs on the test error
            #comp_degreeRem = computeComplimentarity(outputPatSub,outputPatToRem,self.theta);
            comp_overlapRem = CCsolutionSelect(EnOutputPat,outputPatToRem,self.testset['OUT'],self.theta);
            
            #mutate member of the solution to get another possible member
            if self.do_ensemble_mutate :
                mut_sol = self.ensemble_mutate();
            
            #perform cross over of member of the solution to get another possible member
            co_sol = None; #self.ensemble_crossover();
            
            #evaluate with mutant sol
            #(1)add mutant sol to ensemble and then evaluate
            if mut_sol != None:
                
                if self.debug or self.verbose or True:
                    print ">>Using Mutant Solution";
                
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(mut_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(mut_sol);
                    
               
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errMut,outputPatMut = net.evaluate(self.testset,None,True);
                
                #delete
                del net;
                
                #comput complimentarity of the outputs on the test error
                #comp_degreeMut = computeComplimentarity(outputPatSub,outputPatMut,self.theta);
                comp_overlapMut = CCsolutionSelect(EnOutputPat,outputPatMut,self.testset['OUT'],self.theta);
                
                if self.debug or self.verbose or True:
                    print "comp_degree(mutant candidate)", comp_overlapMut;
                
                
                #add solution if its complimentarity and error are better
                if  comp_overlapMut > self.curr_spread  and errMut < errToRem:
                    #add mutant 
                    self.ensemble_sols.append(mut_sol);
                    # remove the solution in question
                    self.ensemble_sols.remove(to_remove_sol);
                    
                    if self.debug or self.verbose :
                        print ">> solution replaced by mutant solution. "
                    
           
            if mut_sol == None:
        
                #pick a solution as replacement solution from the population(top half)
                replacement_i = int(numpy.random.rand() * len(self.sols));            
                replacement_sol = self.sols[replacement_i];
                
                #RECREATE SUBJECT SOLUTION AND EVALUATE ON TEST SET TO GET OUTPUT PATTERN
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(replacement_sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(replacement_sol);
                    
               
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                    
                #evaluate on the test set
                errCompCand,outputPatCompCand = net.evaluate(self.testset,None,True);
                
                #delete
                del net;
                
                #comput complimentarity of the outputs on the test error
                #comp_degreeCand = computeComplimentarity(outputPatSub,outputPatCompCand,self.theta);
                comp_overlapCand = CCsolutionSelect(EnOutputPat,outputPatCompCand,self.testset['OUT'],self.theta);
                
                if self.debug or self.verbose :
                    print "comp_degree(population candidate)", comp_overlapCand;
                
                
                #add solution to ensemble
                if  comp_overlapCand > self.curr_spread  and errCompCand <  errToRem:
                    #add the solution
                    self.ensemble_sols.append(replacement_sol);
                    # remove the solution in question
                    self.ensemble_sols.remove(to_remove_sol);
                    #debug
                    if self.debug or self.verbose:
                        print ">>Replacement Picked from population.";
                    
              
                
    def ensemble_mutate(self,sols = None):
        """ mutates a member of the ensemble"""
        
        if sols == None:
            sols = self.ensemble_sols;
        
        mut_sol = self.mutate(self.ensemble_sols);
        if mut_sol != None:
            return mut_sol;
        else:
            return None;
        
    def ensemble_crossover(self,sols = None):
        """ cross over two solutions an ensemble """
        
        if sols == None:
            sols = self.ensemble_sols;
        
        co_sol = self.crossOver(self.ensemble_sols);
        if co_sol != None:
            return co_sol;
        else:
            return None;
        
    def ensemble_net_evaluate(self,data,ensemble_sols=None):
        """ evaluates the networks of the ensemble """
        
        #select solutions
        if ensemble_sols == None:
            ensemble_sols = self.ensemble_sols;
        
        
        #init 
        enErr = 0.5;
        enVar = 0.0;
        
        #form a cache for keeping the outputs of all the errors of each individual network
        num_compononents = self.ensembles_size;
        dataset_size = len(data['OUT']);
        errors = [];
        
        ensembles_output = numpy.zeros((dataset_size,num_compononents)).astype(numpy.float32);
        ensembles_error = numpy.zeros((dataset_size,num_compononents)).astype(numpy.float32);
        ensembleNet_out = numpy.zeros((dataset_size)).astype(numpy.float32);
        
        ensemble_nets = [];
        ensemble_nets_fitness = [];
        
        if self.debug == True:
            print "Num Of Selected Sols:", len(ensemble_sols);
        
        """ RECREATE SOLUTIONS """
        #create solutions of the ensembles
        for i,x in enumerate(ensemble_sols):
            
            try:
                #create solutions
                sol = ensemble_sols[i];
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(sol);
                
                if self.debug == True:
                    print ">>SOL TO EVAL:", sol;
                    print "genome", sol_genome;
                
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                
                #add ensembles network
                ensemble_nets.append(net);
                ensemble_nets_fitness.append(1-net.fitness);
            except:
                print "-Error Creating networks of ensemble";
                
        """ GET OUTPUTS OF THE SOLUTIONS IN THE ENSEMBLE """
        #evaluate solutions in the ensembles for each pattern
        #if evaluate_multiple == False:
        
        for pati,actual_output in enumerate(data['OUT']):
            
            try:
                if self.debug == True:
                    print data['IN'][pati];
                    print data['OUT'][pati];
                #skip empty pattern
                if data['IN'][pati] == []:
                    continue;
                    
                #get inputs and outputs
                inputs = data['IN'][pati];
                
                
                #stimulate all individual network of the ensemble with the current pattern
                for neti,net in enumerate(ensemble_nets):
                    
                    #stimulate and get outputs
                    out = net.stimulate(inputs);
                    #store ouput
                    ensembles_output[pati][neti] =  max(out);
                    #calculate and store error
                    ensembles_error[pati][neti] = math.pow((max(out) - actual_output),2);

            except:
                print "-Error while evaluating ensembles";
                    
        """ CALCULATE THE DISAGREEMENT AND COMBINE OUTPUTS TO FORM ENSEMBLE OUTPUTS  """
        #calculate disagreement over each pattern (and also do combination)
        for pat_outi,pat_output in enumerate(ensembles_output):
                
            #-ENSEMBLE DISAGREEMENT
            enVar += ensembles_output[pat_outi].var();
            
            #-ENSEMBLE COMBINATION
            #output selection 
            en_outputs = self.combineOutputs(ensembles_output[pat_outi]);
            ensembleNet_out[pat_outi] = max([en_outputs['outAvg'],en_outputs['outMin'],en_outputs['outMax'],en_outputs['outProd']]);
            
    
        #debug/verbose
        if self.debug or self.verbose:
            print "ENSEMBLE OUTPUT", ensembles_output;
            
        """EVALUATE THE OUTPUTS OF THE ENSEMBLE"""
        #Error 
        err = 0.0;
        #calculate mean squared error
        for i, target in enumerate(data['OUT']):
            if target == []:
                continue;
            err += math.pow((ensembleNet_out[i] - (target)),2);
            errors.append(e);
            
        err = err/dataset_size;
        
        #store error
        enErr = (err);
            
        #average of output variances over all patterns
        enVar = enVar/float(len(ensembles_output));
        
        if self.verbose == True:
            print "Ensembles Average disagreement:",self.ensembles_outVar;
            print "Ensembles Error :",self.ensemble_error;
        
        return enErr,enVar;
    
    
        
    def ensemble_evaluate(self, data, ensemble_sols = None):
        """ evaluates the solutions selected as components of ensembles and combines their output to give the ensemble error """
        
        #select solutions
        if ensemble_sols == None:
            ensemble_sols = self.ensemble_sols;
        
        #train on multiple data set
        evaluate_multiple = False;
        
        #check if the training set is more than one
        if (len(data) > 1):
            evaluate_multiple = True;
        
        
        #init output variance
        self.ensembles_outVar = 0.0;
        #form a cache for keeping the outputs of all the errors of each individual network
        num_compononents = self.ensembles_size;
        dataset_size = len(data['OUT']);
        errors = [];
        
        ensembles_output = numpy.zeros((dataset_size,num_compononents)).astype(numpy.float32);
        ensembles_pattern_err = numpy.ones((dataset_size,num_compononents)).astype(numpy.float32);
        ensembles_error = numpy.zeros((num_compononents)).astype(numpy.float32);
        ensembleNet_out = numpy.zeros((dataset_size)).astype(numpy.float32);
        ensembleNet_err = numpy.zeros((dataset_size)).astype(numpy.float32);
        
        ensemble_nets = [];
        ensemble_nets_fitness = [];
        
        if self.debug == True:
            print "Num Of Selected Sols:", len(ensemble_sols);
        
        """ RECREATE SOLUTIONS AND EVALUATE ON PATTERNS"""
        err = 0.0;
        #create solutions of the ensembles
        for i,x in enumerate(ensemble_sols):
            
            try:
                #CREATE THE NETWORK
                sol = ensemble_sols[i];
                #get the genome of the solution
                if self.use_cardinal:
                    sol_genome = decodeGenes(sol,self.useNodeFns,self.useWeightFns);
                else:
                    sol_genome = decodeGenes(sol);
                
                if self.debug == True:
                    print ">>SOL TO EVAL:", sol;
                    print "genome", sol_genome;
                
                #create temp net
                net = ndm();
                #recreate network with the genome
                net.recreateNet(sol_genome);
                
                #EVALUATE THE NETWORK
                for pati,actual_output in enumerate(data['OUT']):
            
                    try:
                        if self.debug == True:
                            print data['IN'][pati];
                            print data['OUT'][pati];
                            
                        #get inputs and outputs
                        inputs = data['IN'][pati];
                        
                        #skip empty inputs
                        if inputs == []:
                            continue;
                        
                        #stimulate and get outputs
                        out = net.stimulate(inputs);
                        #store ouput
                        ensembles_output[pati][i] = max(out);
                        err = math.pow((max(out) - actual_output),2);
                        
                        #error on pattern
                        ensembles_pattern_err[pati][i] = err;
                        
                        #calculate and store error
                        ensembles_error[i] += err * 1.0/dataset_size;
                        
                    except:
                        print "-Error while evaluating ensembles"
                        
                #ADD PHENOTYPE OF ENSEMBLES TO ENSEMBLE NETS
                ensemble_nets.append(net);
                #add ensembles fitness
                ensemble_nets_fitness.append(1-net.fitness);
                
            except:
                print "-Error Creating networks of ensemble";
                
        """ STORE OUTPUT OF THE BEST INDIVIDUAL IN THE ENSEMBLE """
        #Get the best error within the ensemble
        self.best_indErr = min(ensembles_error);
        
        #debug/verbose
        if self.debug or self.verbose:
            print ">>Ensembles Error:", ensembles_error;
            #print ">>Ensemble outputs",ensembles_output;
        
        """ CALCULATE THE DISAGREEMENT AND COMBINE OUTPUTS TO FORM ENSEMBLE OUTPUTS  """
        #calculate disagreement over each pattern (and also do combination)
        for pat_outi,pat_output in enumerate(ensembles_output):
                
            #-ENSEMBLE DISAGREEMENT
            self.ensembles_outVar += ensembles_output[pat_outi].var();
            
            #-ENSEMBLE COMBINATION
            #output selection 
            en_outputs = self.combineOutputs(ensembles_output[pat_outi]);
            #Chose final output - combined by output averaging
            ensembleNet_out[pat_outi] = en_outputs['outAvg'];
            
            #debug
            if self.debug == True:    
                print en_outputs;
        
        """EVALUATE THE OUTPUTS OF THE ENSEMBLE"""
        #Error 
        Err = 0.0;
        err = 0.0;
        #calculate mean squared error
        for i, target in enumerate(data['OUT']):
            #skip empty pattern
            if target == []:
                continue;
            err = math.pow((ensembleNet_out[i] - target),2);
            Err += err;
            #store error
            ensembleNet_err[i] = err;
            
        Err = Err/dataset_size;
        
        #store error
        self.ensemble_error = Err;
            
        #average of output variances over all patterns
        self.ensembles_outVar = self.ensembles_outVar/float(len(ensembles_output));
          
        
        if self.verbose or self.debug == True:
            print "Ensembles Average disagreement:",self.ensembles_outVar;
            print "Ensembles Error :",self.ensemble_error;
        
        return self.ensemble_error,self.ensembles_outVar;
    
    
    def evaluate_multiprocessor(self, data,use_gauss_noise = False, pop = None):
        """ evaluates multiple solutions on different threads to speed up processing """
        #check if there are solutions
        if(pop == None):
            pop = self.sols;
            
        NUM_THREADS = 10;
        threads = [];
        
        
        step = int(len(pop)/float(NUM_THREADS));
        start = 0;
        for i in xrange(NUM_THREADS):
            t = threading.Thread(target=self.evaluate,args=(data,use_gauss_noise,pop[start:start+step]));
            t.setDaemon(True);
            threads.append(t);
            start += step;
            
        for th in threads:
            th.start();
            th.join();
        
    
        
        
    def evaluate(self, data,use_gauss_noise = False, pop = None):
        """ evaluates a population of solutions"""
        
        #check if there are solutions
        if(pop == None):
            pop = self.sols;
            
    
        
        for i,sol in enumerate(pop):
                
            try:
              
                #only evaluate less older solutions    
                if (sol[constants.COST_GENE] == constants.NOT_EVALUATED ) :
                    #get the genome of the solution
                    if self.use_cardinal:
                        #print ">>Cardinal Use - Evaluation"
                        sol_genome = decodeGenes(sol,self.useNodeFns,self.useWeightFns);
                    else:
                        sol_genome = decodeGenes(sol);
                    
                    if self.debug == True:
                        print ">>SOL TO EVAL:", sol;
                        print "genome", sol_genome;
                    
                    #create temp net
                    net = ndm();
                    #recreate network with the genome
                    net.recreateNet(sol_genome);
                    
                    #evaluate on given problem
                    err = max(net.evaluate(data));
                    
                    #evaluate on given problem but with guass noise treatment
                    err2 = max(net.evaluate(data,None,False,True));
                    
                    #include ensemble error
                    #if self.use_ensembles == True:
                    #     err = (err) + ((abs(self.ensemble_error ))  * self.beta);
                    #
                    #assign err
                    sol[constants.COST_GENE] = err;
                    sol[constants.COST2_GENE] = err2;
                            
                    
                    if not self.lock.acquire(False):
                        pass;
                    else:
                        try:
                            
                            #save best network
                            if err < self.best_cost:
                                self.best_sol = list(sol);
                                
                            #save best network
                            if err2 < self.best_cost_test:
                                self.best_sol_test = list(sol);
                                
                            #save best and worst cost
                            self.best_cost = min(err,self.best_cost);
                            self.worst_cost = max(err,self.worst_cost);
                            
                            #SECOND OBJECTIVE
                            #is best cost /worst cost
                            self.best_cost_test = min(err2,self.best_cost_test);
                            self.worst_cost_test = max(err2,self.best_cost_test);
                            
                        finally:
                            self.lock.release();

                    #delete net
                    del net;

            except:
                print "Error while evaluating- solution will be deleted";
                pop.pop(i);
                print traceback.print_exc();

    
    def validate(self,data,All=False,pop = None):
        """ validates the solution on the test data"""
        #check if there are solutions
        if(pop == None):
            pop = self.sols;
        
        best_err = 1.0;
        worst_err = 1.0;
        
        
        for i,sol in enumerate(pop):

            try:
                
                if (sol[constants.COST2_GENE] == constants.NOT_EVALUATED) or (All) :
                    #debug/verbose
                    if self.debug:
                        print ">>SOL CHROMOSOME:", sol;
                        
                    #get the genome of the solution
                    if self.use_cardinal:
                        sol_genome = decodeGenes(sol,self.useNodeFns,self.useWeightFns);
                    else:
                        sol_genome = decodeGenes(sol);
                    
                    if self.debug:
                        print ">>SOL GENOME:", sol;
                        print "genome", sol_genome;
                    
                    #create temp net
                    net = ndm();
                    #recreate network with the genome
                    net.recreateNet(sol_genome);
                    
                    #evaluate on given problem
                    err = max(net.evaluate(data));
                    
                    ##include ensemble error
                    #if self.use_ensembles == True:
                    #         err = 0.5*(err) + ((abs(self.ensemble_error - self.ensembles_outVar))  * self.beta);
                    
                    #assign err
                    sol[constants.COST2_GENE] = err;
                    
                    
                    #is best cost /worst cost
                    best_err = min(err,best_err);
                    worst_err = max(err,worst_err);
                    
                
                    #save best network
                    if err < self.best_cost_test:
                        self.best_sol_test = list(sol);
                        
                        
                    self.best_cost_test = min(err,self.best_cost_test);
                    self.worst_cost_test= max(err,self.worst_cost_test);
                        
                    #delete net
                    del net;
                
            except:
                print "Error while validating - solution will be deleted";
                pop.pop(i);
                print traceback.print_exc();
        
        #return the best error
        return best_err,worst_err;
    
    
    def trimSolutions_multithread(self, pop = None):
        """ trim solutions in parrallel"""
        NUM_THREADS = 10;
        threads = [];
        if pop == None:
            pop = self.sols;
        
        step = int(len(pop)/float(NUM_THREADS));
        start = 0;
        for i in xrange(NUM_THREADS):
            t = threading.Thread(target=self.trim,args=([pop[start:start+step]]));
            t.setDaemon(True);
            threads.append(t);
            start += step;
            
        for th in threads:
            th.start();
            th.join();
        

    def trim(self, pop = None):
        """ trims unwanted solutions and returns the better solutions"""

        if(pop == None):
            pop = (self.sols);

        #debug
        if self.debug or self.verbose :
            print "#TRIMMING SOLUTIONS";
            print "-number to trim:", num_to_trim;

        for i in xrange(len(pop)):
            #select sol
            sol = pop[i];
            
            if(sol[constants.COST_GENE] != -1 and \
                   sol[constants.COST2_GENE] != -1 and \
                   sol[constants.AGE_GENE] > self.min_age_elim and \
                   sol[constants.COST_GENE] >= self.min_cost_elim or\
                   sol[constants.COST2_GENE] >= self.min_cost_elim):
                
                        #eliminitate solution
                        sol_indx = self.sols.index(pop.pop(i));
                        del self.sols[sol_indx];
            
                        
            
    
    def trimSolutions(self, pop = None):
        """ reduces the number of solutions """

        print "-trimming solutions."
     
        if(pop == None):
            pop = (self.sols);
            
        num_sols = len(pop);
        num_to_trim = round(self.percent_trim * num_sols -1);
        
        #debug
        if self.debug or self.verbose :
            print "#TRIMMING SOLUTIONS";
            print "-number to trim:", num_to_trim;
       
        
        for i in xrange(int(num_to_trim)):
            rnd = numpy.random.rand();
            sel_sol, soli = self.randomSelection(1);
            sol = sel_sol[0];
            #add to nwlist only if the random number is greater than prob of elim.
            if(rnd >= self.prob_elim):
                if(sol[constants.COST_GENE] != -1 and \
                   sol[constants.COST2_GENE] != -1 and \
                   sol[constants.AGE_GENE] > self.min_age_elim and \
                   sol[constants.COST_GENE] >= self.min_cost_elim or\
                   sol[constants.COST2_GENE] >= self.min_cost_elim):
                    #eliminitate solution
                    del pop[soli.pop()];
                    
                    
        if len(self.sols) > 150:
            self.sortSolutions();
            del self.sols[150:];
 
    #NOTE : works, but needs implementation tweaks to work with variable length solutions
    def gpu_ageSols(self, pop = None):
        """ makes population of solution to age (operation done on gpu in parallel) """
        
        print "-gpu aging solutions";
        
        if(pop == None):
            pop = self.sols;

        cudaInterface.cuda_ageSols(pop);
        
        return pop;
        
    def ageSolutions(self, pop = None):
        """ make the solutions to age """

        print "-aging solutions."

        if(pop == None):
            pop = self.sols;

        num_sols = len(pop)-1;
        

        # use GPU to perform this operation
        for soli in xrange(num_sols):
            curr_age = pop[soli][constants.AGE_GENE];
            pop[soli][constants.AGE_GENE] = curr_age +1;

        return pop;

    
    def crossOver(self, pop = None):
        """ performs cross over operation on two operations"""
        if (pop == None):
            pop = self.sols;
            
        #get the total number of solutions
        num_sols = len(pop);
        trialSol = [];
        
        #number of required solutions for operation.
        num_req_sols = 2;
        
        if self.debug == True:
            print"num_sols:", num_sols;
      
        #choose candidate solutions randomly - Note: solutions chosen based on similarity of length
        # TO DO - consider using cross-over selection based on similarities of architecture
        cand_sols,cand_indc = self.randomSelection(num_req_sols);
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
          
        #retrive the two candidates
        if cand_sols != None:
            
            solAi = cand_indc[0];
            solBi = cand_indc[1];
            
            if self.debug == True:
                
                print "parentA:", pop[solAi];
                print "len A:", len(pop[solAi]);
                print "parentB:", pop[solBi];
                print "len B:", len(pop[solBi]);

            #cross over connectivity, weights, autoweight and bias including transfer functions but leave network architecture intact
            crossOverFrom = constants.META_INFO_COUNT + num_nodes ;
            
            #get solutions
            solA = pop[(solAi)][crossOverFrom:];
            solB = pop[(solBi)][crossOverFrom:];
    
            #prob of crossover
            prCO = numpy.random.rand();
    
            if(prCO <= self.prob_cross_over ):
                #perform cross-over
                pointOfCrossOver = round(numpy.random.rand() * (len(solA)-1));
                #put in range for effective cross-over
                if pointOfCrossOver < 0:
                    pointOfCrossOver = 1;
                elif pointOfCrossOver > len(solA):
                    pointOfCrossOver = len(solA)-1;
    
                if self.debug == True:
                    print "\n -performing crossover";
                    print "\n -parent A:", solA;
                    print "\n -parent B:", solB;
                
    
                #tranfer genetic material
                #copy architecture of solA
                trialSol = pop[solAi][constants.META_INFO_COUNT : crossOverFrom];
                #copy the genes of solutionA
                nwGenes = (solA);
                #cross over the genes
                nwGenes[int(pointOfCrossOver+1):] = solB[int(pointOfCrossOver+1):];
                #add cross-over genes to offpsring
                trialSol.extend(nwGenes);
    
                #set as unevaluated
                cost = -1;
                age = 0;
                misc = 0;
                
                trialSol.insert(constants.COST_GENE,cost);
                trialSol.insert(constants.COST2_GENE,cost);
                trialSol.insert(constants.MISC_GENE,misc);
                trialSol.insert(constants.AGE_GENE,age);
                
                if self.verbose == True:
                    print "\n -trail solution:", trialSol
                
    
            else:
                print "\n no crossOver";
                return None;
            
        else:
            #no similar solutions found suitable for crossover
            print "\n no crossover"
            return None;

        #return
        return trialSol;
    
    def probMingle(self, solA, solB):
        """ probablistic mingling of solutions based on fitness"""
        pass;

    def gpu_mutate(self, pop = None):
        """ performs mutation on population of solutions """
        if(pop == None):
            pop = self.sols;
            
        #one solution   
        num_req_sols = 5;
        
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        #mutate not on architecture
        mutateFrom = constants.META_INFO_COUNT + num_nodes;
        
        #select one solution at random
        sols,solsi = self.randomSelection(num_req_sols);
        
        #check if there are any solutions
        if sols == None and solsi == None:
            return None;
        
        #only work with minimum number of solution to prevent transfer overhead
        if len(sols) < num_req_sols:
            return None;
        
        #conver to numpy array
        to_mut_sols = numpy.array(sols,numpy.float32);
        

        #mutate solutions on gpu
        mut_sols = cudaInterface.cuda_mutate(to_mut_sols,self.prob_mutation,self.mutation_range,self.param_min,self.param_max);
        
            
        if self.debug == True:
            print "mut_sols:", mut_sols;
        
        #return new sols
        return mut_sols;
            
        
    
    def gauss_mutate(self,pop=None):
        """ does a mutation from values sampled out of a gaussian distrubution """
        if(pop == None):
            pop = self.sols;
            
        req_num_sol = 1;
        
        
        #distribution settings
        mean = self.gaus_mut_mean;
        std_dev = self.gaus_mut_std;
    

        minVal = self.param_min;
        maxVal = self.param_max;
        mutRange =  2 * self.mutation_range;
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        #mutate not on architecture
        mutateFrom = constants.META_INFO_COUNT + num_nodes;
        
        #mutant sol
        nwSol = [];
        
        #select ranomd solution
        sols,solsi = self.randomSelection(req_num_sol);
        #get solution
        sol = sols.pop();
        #copy
        nwSol = list(sol);
        
        if self.debug:
            print "pre_mutated sol:", nwSol;
        
        #mutation values
        vals = numpy.random.normal(mean, std_dev, len(sol)-mutateFrom);
                
        #mutate        
        nwSol[mutateFrom:] += vals;
        
        #assign cost and age
        nwSol[constants.COST_GENE] = -1.0;
        nwSol[constants.COST2_GENE] = -1.0;
        nwSol[constants.MISC_GENE] = 0.0;
        nwSol[constants.AGE_GENE] = 0;
        
        if self.debug :
            print "post_mutatated sol:", nwSol;

        
        #put within range
        #nwSol = [putInRange(x,minVal,maxVal) for x in nwSol[mutateFrom:]];
        
        #return solution
        return nwSol;
        
        
        
    def mutate(self, pop = None):
        """mutates the from the population of solutions"""
        if(pop == None):
            pop = self.sols;
            
        #mutates the solution
        vec_len = 0;
        req_num_sol = 1;
       
        cost_and_age_genes  = constants.META_INFO_COUNT;

        minVal = self.param_min;
        maxVal = self.param_max;
        mutRange =  2 * self.mutation_range;
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        #mutate not on architecture
        mutateFrom = constants.META_INFO_COUNT + num_nodes;
        
        #mutant sol
        nwSol = [];
        
        #select ranomd solution
        sols,solsi = self.randomSelection(req_num_sol);

        if sols != None:
            
            #get solution
            sol = sols.pop();
            
            #mutant solution - copies architecture of same network
            nwSol = sol[constants.META_INFO_COUNT : mutateFrom];
            
            if self.debug == True:
                print "Origninal Solution:", sol;
            
            #MUTATE - prob. that determines whether to mutate or not
            prM = numpy.random.rand(); #current probability of mutation 
            
            if(prM <= self.prob_mutation):
    
                if self.debug == True:
                    print "\n -performing mutation";
                    print "\n -parent:", sol;
    
                for g in sol[mutateFrom:] :
                    #generate ranomd number
                    prM = numpy.random.rand();
                    
                    
                    if (prM <= self.prob_mutation):
                        #calculate mutation value
                        mutVal = (numpy.random.rand() * mutRange) - self.mutation_range;
                        #mutate genes
                        nwSol.append(g + mutVal);
                    else:
                        nwSol.append(g);
                
                if self.debug == True:
                    print "\n- mutant sol:", nwSol;
    
            else:
                print "\n -No Mutation";
                return None;
            
            
            cost = -1;
            age = 0;
            misc =0.0;
            #append cost and age
            nwSol.insert(constants.COST_GENE,cost);
            nwSol.insert(constants.COST2_GENE,cost);
            nwSol.insert(constants.MISC_GENE,misc);
            nwSol.insert(constants.AGE_GENE,age);
            
            if self.debug == True:
                print "Final solution";
                print "len:", len(nwSol);
                print "genes", nwSol;
        
        #return sol
        return nwSol;
    
    def mutateArchitecture(self, pop = None):
        """mutates the architecture of solutions"""
        
        if(pop == None):
            pop = self.sols;
            
        #mutates the solution
        vec_len = 0;
        req_num_sol = 1;
        cost = 0;
        age = 0;
        cost_and_age_genes  = 2;

        minVal = self.param_min;
        maxVal = self.param_max;
        mutRange =  self.mutation_range;
        
        #get number of nodes
        num_nodes = netParams.nodeConfig['I'] + netParams.nodeConfig['H'] + netParams.nodeConfig['O'];
        
        #mutate not on architecture
        mutateTo = constants.META_INFO_COUNT + num_nodes;
        
        #mutant sol
        nwSol = [];
        
        #select ranomd solution
        sols,solsi = self.randomSelection(req_num_sol);
        
        if sols != None:
            
            #get solution
            sol = sols.pop();
            
            #insert cost and age
            nwSol = [-1 ,0];
           
            if self.debug == True:
                print "Origninal Solution:", sol;
            
            #MUTATE - prob. that determines whether to mutate or not
            prM = numpy.random.rand(); #current probability of mutation 
            
            if(prM <= self.prob_mutation):
    
                if self.debug == True:
                    print "\n -performing mutation";
                    print "\n -parent:", sol;
    
                for g in sol[constants.META_INFO_COUNT : mutateTo]:
                    #generate ranomd number
                    prM = numpy.random.rand();
                    
                    
                    if (prM <= self.prob_mutation):
                        
                        #flip
                        g = flipBit(g);
                        
                        #mutate genes
                        nwSol.append(g);
                    else:
                        nwSol.append(g);
                        
                        
                    
                #debug
                if self.debug == True:
                    print "\n- mutant sol:", nwSol;
    
            else:
                if self.debug == True:
                    print "\n -No Mutation";
                    
                return None;
            
            
          
          
            if self.debug == True:
                print "Final solution";
                print "len:", len(nwSol);
                print "genes", nwSol;
        
        #return sol
        return nwSol;
        

    def normalise(self, sol,minVal,maxVal):
        """ normalise the values of the chromosome within the param range given """
        nsol = [putInRange(x,minVal,maxVal) for x in sol];
        return nsol;


    def sortSolutions(self, pop = None):
        """ sorts the solutions according to their fitness """
        if(pop == None):
            pop = self.sols;
        try:
            #sort along the selected axis
            pop.sort(lambda x, y: cmp(x[constants.COST_GENE],y[constants.COST_GENE]));
        except:
            print "problem while sorting";
        #return sorted solutions    
        return pop;
    
    def sortSolutionsByTestError(self, pop = None):
        """ sorts the solutions accordin to their test error """
        if(pop == None):
            pop = self.sols;
            
        try:
            #sort along the selected axis
            pop.sort(lambda x, y: cmp(x[constants.COST2_GENE],y[constants.COST2_GENE]));
        except:
            print "problem while sorting";
        #return sorted solutions    
        return pop;
        

    """ SELECTION SCHEMES """
    def randomSelection(self, num_to_sel, pop = None):
        """ randomly selects number of individuals needed from the population
            @return returns none if there are no matches found 
        """
        
        if self.verbose == True:
            print " -using random selection";
        
        #default
        if(pop == None):
            pop = self.sols;

        #get number of solutions
        num_sols =len(pop);

        #number of nodes
        num_nodes = netParams.nodeConfig['I']  + netParams.nodeConfig['H']  + netParams.nodeConfig['O'] ;
        
        #desired length of solution
        sol_selected_len = 0;

        #tried indices
        tried_indx = [];

        #form list to store selected sols
        selected_sols = []; #selected solutions list
        selected_solsInd = []; #indices of selected solutions
        selected_solArch = []; #architecture of selected sol

        tries = 0;
        max_tries = 10;
        max_attemps = 10;
        solXi = 0;

        for t in xrange(max_attemps):
            
            #select number solutions
            while len(selected_sols) != num_to_sel:

                #select solution randomly
                solXi = int(round(numpy.random.rand() * num_sols -1));
                #architecture of solX
                solXArch = pop[solXi][constants.META_INFO_COUNT : num_nodes];
                
                #get length for finding similar lenghted solutions
                if(len(selected_solsInd) == 0):
                    sol_selected_len = len(pop[solXi]);
                    #debugger
                    if self.debug == True:
                        print "selected len:", sol_selected_len;

               #only accept solutions of similar length
                if(solXi not in selected_solsInd and \
                   len(pop[solXi]) == sol_selected_len):
                    #add index
                    selected_solsInd.append((solXi));
                    #add solution
                    selected_sols.append(pop[solXi]);
                    #sol selected len
                    sol_selected_len = len(pop[solXi]);
                    #sol selected architecture
                    selected_solArch = pop[solXi][constants.META_INFO_COUNT : num_nodes];
                    #add to tried indices
                    tried_indx.append(solXi);

                
                if tries == max_tries:
                    selected_sols = [];
                    selected_solsInd = [];
                    sol_selected_len = 0;
                    tries = 0;
                    #debug
                    if self.debug == True:
                        print "-Retrying."
                        
                    break;
                
                #increament tries
                tries += 1;
                

            
        if len(selected_sols) != num_to_sel:
            return None,None;
        #return selected solutions
        return selected_sols,selected_solsInd ;
    
    def update(self):
        """ """
        global best_cost_curve, data, ptr, p6
        best_cost_curve.setData(self.lst_best_cost);
        
        ptr += 1

    def getObjcosts(self,pop = None):
        """returns the costs for the objectives """
        
        if pop == None:
            pop = self.sols;
        
        f1 = [];
        f2 = [];
        for sol in pop:
            f1.append(sol[constants.COST_GENE]);
            f2.append(sol[constants.COST2_GENE]);
            
            
        return f1,f2;
            
        
    def getBest(self):
        """ returns the refernece of the best cost, and the solution with that cost in the population so far"""
        
        best_cost = float('inf'); #best cost
        best_sol = []; #best solution (reference)
        best_soli = ''; #best solutions index
        
        for soli,sol in enumerate(self.sols):
            cost = sol[constants.COST_GENE];
            if cost < best_cost and cost != constants.NOT_EVALUATED:
                best_cost = cost;
                best_soli = soli;
                best_sol = sol;
        
        #debug/verbose
        if self.debug or self.verbose or True:
            print ">>best cost:", best_cost;
            

        return best_cost,best_soli,best_sol;
        

        
    def getWorst(self):
        """ return the worst cost and the solution with that cost"""
        
        worst_cost = 0.0; #worst cost
        worst_sol = []; #worst solution (reference)
        worst_soli = ''; #worst solutions index
        
        for soli,sol in enumerate(self.sols):
            cost = sol[constants.COST_GENE];
            if cost > worst_cost and cost != constants.NOT_EVALUATED:
                worst_cost = cost;
                worst_soli = soli;
                worst_sol = sol;
        
        #debug/verbose
        if self.debug or self.verbose or True:
            print ">>worst cost:", worst_cost;
            
        
        return worst_cost,worst_soli,worst_sol;
        
    

    #Works, but extremely slow - an alternative for random solutions
    def findAllSimilarSols(self,sol,num_to_sel=None):
        """ find solutions similar to the given solutions in length """
        if self.verbose == True:
            print " -using findAllSimilarSols";
        
        #get population of solutions
        pop = self.sols;
        
        #desired length of solution
        sol_selected_len = len(sol);

        #form list to store selected sols
        selected_sols = []; #selected solutions list
        selected_solsInd = []; #indices of selected solutions
  
        #scan through populations for similar lengthened solutions
        for i,x in enumerate(pop):
            if len(x) == sol_selected_len:
                selected_sols.append(x);
                selected_solsInd.append(i);
                
            if num_to_sel != None:
                if len(selected_sols) == num_to_sel:
                    break;

        #return selected solutions
        return selected_sols,selected_solsInd ;

    
    """ DEBUG METHODS FOR OPTIM """
    def print_popInfo(self,pop= None):
        """prints the lengths of the solutions """
        
        if pop == None:
            pop = self.sols;
        
        print ">>>POP INFORMATION";
        print "Pop Size:", len(self.sols);
        
        for i,x in enumerate(pop):
            print "CHROMOSOME#",i;
            print "-len:", len(x);
            print "-cost:", x[constants.COST_GENE];
            print "-age:", x[constants.AGE_GENE];
    
