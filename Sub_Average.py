# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 20:49:57 2014
@author: Brenton Mallen

This code is used to average multiple output files for the Kaggle Shoppers Challenge

inputs: file directiory, save file name and the submission filenames as strings with extension
"""

#%% Define the submission file directiories
import numpy as np
import csv

def ShopOutAve(directory, saveName, *arg): # arg is meant to be the file names as strings
    
    if len(arg) > 1:
        Mat = np.genfromtxt(directory + arg[0], delimiter = ',') # get the first file
        Labels = Mat[1:,0] # get the labels for later
        Labels = Labels.astype(str) # convert the labels from float to string
        for q, val in enumerate(Labels):    # loop through labels and remove decimal 
            temp = Labels[q]
            Labels[q] = temp[:-2]
        Labels = ['id'] + Labels.tolist()   # convert labels to a list and add the header label


        Data = Mat[1:,1] # get the data that we want to average (second column of the submission files)
        
        for ele in arg[1:]: # loop through each submission file and grab the probability vector
            Mat_temp = np.genfromtxt(SaveDirec + (ele), delimiter = ',')
            Data = np.column_stack((Data,Mat_temp[1:,1]))  # concatenate the probability vectors into one array
    else:
        print('Only one input file. No averaging')
        break

    AveData = np.mean(Data, axis = 1)   # calculate the average across all submission files for each index
    
    
    AveData = ['repeatProbability'] + AveData.tolist() # create list of averages and add the header label

    
# Write the lists to a csv file.
    with open(directory + saveName,'wb') as f:
        for i, val in enumerate(AveData):
            if i == 0:
                f.write(Labels[0] + ','+ AveData[0] + "\n")
            else:
                
                f.write(str(Labels[i])+','+"%10.7f" % AveData[i]+"\n")
            
#%% --------------------- Run the function ----------------------
SaveDirec = 'F:/Shopping/Data/'
fname1 = 'kaggle.submission.csv'
fname2 = 'kaggle.submission_Tushar.csv'
SaveName = 'AveragedSubmission.csv'
ShopOutAve(SaveDirec,SaveName,fname1,fname2)



