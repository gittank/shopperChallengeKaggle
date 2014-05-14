# -*- coding: utf-8 -*-
"""
Created on Wed May 07 15:50:49 2014

@author: miguel.rufino
"""

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def makeFeatureArray(loc_train):
    
    import numpy as np
    
    N = file_len(loc_train)    # number of examples
    M = 56                 # number of features    

    # names of all features
    name_feat = ["never_bought_company","never_bought_category","never_bought_brand","has_bought_brand_company_category","has_bought_brand_category",\
                "has_bought_category_company","has_bought_brand_company","offer_value","market","chain","total_spend",\
                "has_bought_company","has_bought_company_q","has_bought_company_a","has_bought_company_30","has_bought_company_q_30","has_bought_company_a_30",\
                "has_bought_company_60","has_bought_company_q_60","has_bought_company_a_60","has_bought_company_90","has_bought_company_q_90","has_bought_company_a_90",\
                "has_bought_company_180","has_bought_company_q_180","has_bought_company_a_180",\
                "has_bought_category","has_bought_category_q","has_bought_category_a","has_bought_category_30","has_bought_category_q_30","has_bought_category_a_30",\
                "has_bought_category_60","has_bought_category_q_60","has_bought_category_a_60","has_bought_category_90","has_bought_category_q_90","has_bought_category_a_90",\
                "has_bought_category_180","has_bought_category_q_180","has_bought_category_a_180",\
                "has_bought_brand","has_bought_brand_q","has_bought_brand_a","has_bought_brand_30","has_bought_brand_q_30","has_bought_brand_a_30",\
                "has_bought_brand_60","has_bought_brand_q_60","has_bought_brand_a_60","has_bought_brand_90","has_bought_brand_q_90","has_bought_brand_a_90",\
                "has_bought_brand_180","has_bought_brand_q_180","has_bought_brand_a_180"]
    
    # indices of all features
    index_feat = np.arange(0,M,1)        

    dict_feat = dict(zip(name_feat,index_feat))     # create a dictionary of the features & indices
    
    features = np.zeros((N,M))                # create NxM array of zeros for features
    labels   = np.zeros((N,1))                # create Nx1 array of zeros for labels    
    ids   = np.zeros((N,1))                # create Nx1 array of zeros for labels    
    
    # load data from file   
    for n, line in enumerate( open(loc_train) ):
        print '.',            
        row = line.strip().split(" ")        # extract row elements
        for e, el in enumerate(row):            
            item = row[e].strip().split(":")     # split elements into name and value
            if e==0:
                labels[n] = item            
            if e==1:
                ids[n]    = item[0][1:]
            
            if e>2:
                featureName  = item[0]
                featureValue = item[1]
                # find key in dictionary - value is location to store at        
                features[n,dict_feat[featureName]] = featureValue
        if n % 1000 == 0:
            print n
    print '\n Design Array Created \n\n'    
    return features,labels,ids
    
    
def makeSubmission(loc_submission,predictions,ids):
    with open(loc_submission, "wb") as outfile:
        for e in range(np.shape(predictions)[0]):
            outfile.write(str(predictions[e]) + " " + str(int(ids[e])) + "\n")
            if e % 100 == 0:
                print e
    
if __name__ == '__main__':
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    from sklearn import cross_validation
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as pl  
    
    classifier        = 'RF'
    crossval          = 1
    loc_train         = 'C:/Users/tushar.tank/Documents/GitHub/3PiShopper/Data/train.vw'
    loc_test          = 'C:/Users/tushar.tank/Documents/GitHub/3PiShopper/Data/test.vw'
    loc_submission    = 'C:/Users/tushar.tank/Documents/GitHub/3PiShopper/Data/treePreds.txt' 
    NUM_EST           = 500;
    
    features,labels,ids             = makeFeatureArray(loc_train)           # extract train features 
    testFeatures,testLabels,testids     = makeFeatureArray(loc_test)        # extract test features

    rmf = RandomForestClassifier(n_estimators=NUM_EST, oob_score=True, n_jobs=3).fit(features, np.ravel(labels))
    
    # test data set for submission
    rmfTest = rmf.predict_proba(testFeatures)
    makeSubmission(loc_submission, rmfTest[:,1], testids)
"""    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, np.ravel(labels), test_size=0.01, random_state=0)

    
    
    rmf = RandomForestClassifier(n_estimators=NUM_EST, oob_score=True, n_jobs=3).fit(X_train, y_train)
    # predict on training data just for testing
    ypTrain = rmf.predict(X_train)
    print 'OOB score: %.2f\n' % rmf.oob_score_
    print metrics.confusion_matrix(ypTrain, y_train)
    
    ypTest = rmf.predict(X_test)
    rmfTestProb = rmf.predict_proba(X_test)
    print 'OOB score: %.2f\n' % rmf.oob_score_
    print metrics.confusion_matrix(ypTest, y_test)
    print '\nFeature Importance'
    print rmf.feature_importances_
    
    
    fpr, tpr, thresholds = roc_curve(y_test, rmfTestProb[:,1], pos_label=1) 
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig = pl.figure(figsize=(12,8))
    # We define a fake subplot that is in fact only the plot.  
    plot = fig.add_subplot(111)
    # We change the fontsize of minor ticks label 
    plot.tick_params(axis='both', which='major', labelsize=18)
    
    pl.plot(fpr, tpr, marker='*',label='ROC curve (area = %0.4f)' % roc_auc, linewidth=1)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    
    pl.xlabel('False Positive Rate ($P_f$)',fontsize=24)
    pl.ylabel('True Positive Rate ($P_d$)',fontsize=24)
    pl.title('Number of Estimates %d' % NUM_EST, fontsize=32)
    pl.legend(loc="lower right")
    pl.grid()
    pl.show()
"""
    

    
    
    
    
    