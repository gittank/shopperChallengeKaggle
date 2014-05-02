# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 14:58:53 2014

@author: miguel.rufino
"""

# collapse product measure and quantity into single value
# purchase cost per unit 

import json as json

# build dictionary of categories and departments
def genCategoryDict(inName,outName):
    a={}
    for count, l in enumerate(open(inName)):
#    infile = open(inName)
#    line = infile.readlines(10000);           
#    for count,l in enumerate(line):    
        if count>1: # skip header
            department = l.split(",")[2]            # extract department
            category  = l.split(",")[3]             # extract category
            if department in a:                     # if department already in dictionary  
                if category not in a[department]:   # if it's not a duplicate
                    a[department].append(category)  # add categories to list
            else:
                a[department] =  [category];        # update dictionary
                       
            if count % 5000000 == 0:                # track progress           
                print count
                
    with open(outName, 'w') as f:
        json.dump(a, f)


def addDepartmentToOffers(dictName,offers_loc,outName):
    
    # load dictionary from file    
    with open(dictName) as f:
        my_dict = json.load(f) 
        
    # for read every line
    for count, l in enumerate(open(offers_loc)):
        if count == 0:
            with open(outName, 'a') as f:            
                f.write('%s,department\n' %(l[:-1]) )  # write header to file
        else: # skip header
            category    = l.split(",")[1]                   # extract offer category
            
            # what department does it belong to
            for i,values in enumerate(my_dict.values()):
                if category in values:
                    department = my_dict.keys()[i]
                    break
           
            with open(outName, 'a') as f:            
                f.write('%s,%d\n' %(l[:-1],int(department)))        # write line to file      
        


if __name__ == '__main__':
    
    trans_loc = 'transactions.csv'
    dict_loc  = 'catToDeptDict.json'    
    offers_loc= 'offers.csv'
    outName   = 'offersDept.csv'
    
    # genCategoryDict(trans_loc, dict_loc)
    addDepartmentToOffers(dict_loc, offers_loc,outName)    	
 
