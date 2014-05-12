# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:08:29 2014

@author: Brenton.Mallen

This script is intended to facilitate a grid search for Vowpal Wabbit for the
Kaggle Shopping competition.

This code will have 2 main parts. Part 1: Generate the command line code to run
VW for both quantile and square loss functions -  loop through variables. Part
2: determine performace for each itteration.

"""

#-------------------------------------------------------------------------
def gen_DataSplit(Training, TrainLabels,Test_Split_out, Train_Split_out, Train_Split_Label, Test_Split_Label, testSize):
    ''' This function is used to split the training data into two subsets to use for validation
        It goes through the full training data and slits up based on "testSize" input.
        It also creates label files for each subset - this is used for auc calculation later.
        These are all saved to separate files.

        Training - the input training data set
        TrainLabels - vector of the labels
        Test_Split_out, Train_Split_out, Train_Split_Label, Test_Split_Label - the output file names
        '''
    from numpy import floor
    for e,line in enumerate(open(Training)): # get number of rows in training set
        pass
    numRows = e + 1
    testNum = floor(numRows * testSize)

    # Create Training subset and corresponding labels
    with open(Train_Split_out,'wb') as TTrain: 
        with open(Train_Split_Label, 'wb') as TTrainL:
            for i,l in enumerate(open(Training)):
                if i > testNum:
                    TTrain.write(l)
                    TTrainL.write(l[0][0] + '\n')
    
    # Create Training subset and corresponding labels            
    with open(Test_Split_out,'wb') as TTest:
        with open(Test_Split_Label,'wb') as TTestL:
            for a,q in enumerate(open(Training)):
                if a <= testNum:
                    TTest.write(q)
                    TTestL.write(q[0][0] + '\n')
                    
#-------------------------------------------------------------------------   


def gen_CommandLine(TrainingData, TestData, Model_Temp, Preds_out, LossChoice,\
 numPasses, LearnRate, Hessian, RandWeights, Tau): 
    '''The function is used to generate the command line text to initiate vowpal.
        The vw parameters are the inputs to the function.

        TrainingData, TestData - input file names as strings
        Model_Temp, Preds_out - strings of the file names or vw outputs
        LossChoice, numPasses, LearnRate, Tau - numerical value inputs
            if using squared loss, set Tau = []
        Hessian, RandWeights - boolean inputs'''
    
    import sys, os    
    #%%-------------------------Square Loss----------------------------------------
    if LossChoice == 'squared':
            # Random Weights Activation
        if RandWeights == 'on':
            RW = ' --random_weights(on) '
        elif RandWeights == 'off':
            RW = ''
        else:
            sys.exit('DANGER Will Robinson!: Invalid argument for Random Weights activation')
        # Hessian Activation
        if Hessian == 'on':
            Hess = 'hessian_on'
        elif Hessian == 'off':
            Hess = ''
        else:
            sys.exit('DANGER Will Robinson!: Invalid argument for Hessian activation')
        
        
        Command_Train = 'vw ' + TrainingData + ' -c -k --passes '+ str(numPasses) +' -l ' + str(LearnRate) + \
        ' -f ' + Model_Temp + ' --loss_function squared' + RW + Hess
        
#        print  'Loss: ' + str(LossChoice) + '# Passes: ' + str(numPasses) + 'Learn Rate: '\
#        + str(LearnRate) + 'Hessian: ' + Hessian + 'Random Weights: ' + RandWeights
        
    #%%------------------------Quantile Loss---------------------------------------    
    elif LossChoice == 'quantile':
        
        # Random Weights Activation
        if RandWeights == 'on':
            RW = ' --random_weights(on) '
        elif RandWeights == 'off':
            RW = ''
        else:
            sys.exit('DANGER Will Robinson!: Invalid argument for Random Weights activation')
        
        # Hessian Activation
        if Hessian == 'on':
            Hess = 'hessian_on'
        elif Hessian == 'off':
            Hess = ''
        else:
            sys.exit('DANGER Will Robinson!: Invalid argument for Hessian activation')
        
    
        Command_Train = 'vw ' + TrainingData + ' -c -k --passes '+ str(numPasses) +' -l ' + str(LearnRate) + \
        ' -f ' + Model_Temp + ' --loss_function quantile --quantile_tau ' + str(Tau) + RW + Hess
        
#        print  'Loss: ' + str(LossChoice) + 'Tau: ' + str(Tau) + '# Passes: ' + str(numPasses) + 'Learn Rate: '\
#        + str(LearnRate) + 'Hessian: ' + Hessian + 'Random Weights: ' + RandWeights
        
    else:
        sys.exit('DANGER Will Robinson!: Invalid argument for Loss Choice')                        
    #%%------------Train & Run on Test Data ----------------------------------
    
    Command_Test = 'vw ' + TestData + ' -t -i ' + Model_Temp + ' -p ' + Preds_out

    os.system(Command_Train)   # send command to terminal 
    os.system(Command_Test)
        
    return Preds_out

'''============================================================================
============                  Validate Models                    ==============
============================================================================'''

def perf_metric(InputLabels, Predictions):
    '''This function is used to calculate the AUC metric.
        InputLabels - the vector of true labels for the training subset 
        Predictions - output from vowpal'''
    import numpy as np
    from sklearn.metrics import roc_auc_score
    
    L = np.loadtxt(InputLabels)
    P = np.loadtxt(Predictions)[:,0]
    AUC = roc_auc_score(L,P)

    return AUC

#%%============================================================================
'''============================================================================
============                   Run on Data                =====================
============================================================================'''
if __name__ == '__main__':
    Directory = '/media/sf_Docs/Kaggle/Shopping/Data/'
    TrainingData = Directory + 'train.vw'
    Test_Split_out = Directory + 'train_Test_Split.vw'
    Train_Split_out = Directory + 'train_Train_Split.vw'
    Test_Split_Label = Directory + 'Test_Split_Labels.vw'
    Train_Split_Label = Directory + 'Train_Split_Labels.vw'
    
    TrainLabels = Directory + 'ReducedTrain_Labels.vw'
    Preds_out = Directory + 'Preds_out.txt'
    Model_Temp = Directory + 'model_temp.vw'
    
    testSize = 0.6    
    LossChoice = ('squared','quantile') # squared or quaintile
    numPasses = np.arange(50,500,50) # number of passes. Same for each loss function case
    LearnRate = np.arange(0.35,1,0.05) # learning rate. 0.5 = Default
    Hessian = ('off','on') # On or Off as string
    RandWeights = ('off','on')  #  On or Off as string
    Tau = np.arange(0.25,1,0.05) # Quantile tau

    gen_DataSplit(TrainingData, TrainLabels, Test_Split_out, Train_Split_out,Train_Split_Label,\
    Test_Split_Label, testSize)
    
    with open(Directory + 'VW_Results.txt','wb') as outfile:
        for lossC in LossChoice:
            for numP in numPasses:
                for learnR in LearnRate:
                    for hess in Hessian:
                        for weights in RandWeights:
                            if lossC == 'quantile':
                                for t_ow in Tau:
            
                                    Predict = gen_CommandLine(Train_Split_out, Test_Split_out, Model_Temp, Preds_out,\
                                    lossC, numP, learnR, hess, weights, t_ow)
                                    
                                    AUC = round(perf_metric(Test_Split_Label,Predict),8)
                                    
                                   # R = open(Directory + 'VW_Results.txt','wb') as outfile:

                                    
                                    outfile.write ('Loss: ' + lossC + ', Tau: ' + str(t_ow) + ', Passes:' + str(numP) + ', LearnRate: ' + str(learnR) + ', Hessian: '\
                                    + hess + ', RandWeights: ' + weights\
                                    + ', AUC: ' + str(AUC) + '\n')
                            else:
                                Predict = gen_CommandLine(Train_Split_out, Test_Split_out, Model_Temp, Preds_out,\
                                lossC, numP, learnR, hess, weights,[])
                                    
                                AUC = round(perf_metric(Test_Split_Label,Predict),8)
                                
                                
                                outfile.write ('Loss: ' + lossC + ', Passes:' + str(numP) +  ', LearnRate: ' + str(learnR) + ', Hessian: '\
                                + hess + ', RandWeights: ' + weights\
                                + ', AUC: ' + str(AUC) + '\n')







