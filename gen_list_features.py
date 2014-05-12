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
	
	N = file_len(loc_train)	# number of examples
	M = 56 				# number of features	

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

	dict_feat = dict(zip(name_feat,index_feat)) 	# create a dictionary of the features & indices
	
	features = np.zeros((N,M))				# create NxM array of zeros for features
	labels   = np.zeros((N,1))				# create Nx1 array of zeros for labels	
	ids   = np.zeros((N,1))				# create Nx1 array of zeros for labels	
	
	# load data from file   
	for n, line in enumerate( open(loc_train) ):
		print '.',			
		row = line.strip().split(" ")		# extract row elements
		for e, el in enumerate(row):			
			item = row[e].strip().split(":") 	# split elements into name and value
			if e==0:
				labels[n] = item			
			if e==1:
				ids[n]    = item[0][1:]
			
			if e>2:
				featureName  = item[0]
				featureValue = item[1]
				# find key in dictionary - value is location to store at		
				features[n,dict_feat[featureName]] = featureValue
		if n % 10 == 0:
			print n
		
	return features,labels,ids
	
	
def makeSubmission(loc_submission,predictions,ids):
	with open(loc_submission, "wb") as outfile:
		for e in range(np.shape(predictions)[0]):
			outfile.write(str(predictions[e]) + " " + str(int(ids[e])) + "\n")
			if e % 100 == 0:
				print e
	
if __name__ == '__main__':
	
	from sklearn.metrics import roc_curve, roc_auc_score
	from sklearn.cross_validation import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn import linear_model
	import matplotlib.pyplot as plt
	import numpy as np
	
	classifier 		= 'GBR'
	crossval  		= 1;
	loc_train 		= 'train.vw'
	loc_test  		= 'test.vw'
	loc_submission 	= 'treePreds.txt' 
	
#	features,labels,ids 			= makeFeatureArray(loc_train)	# extract train features 
#	testFeatures,testLabels,testids 	= makeFeatureArray(loc_test)		# extract test features
	
	if crossval==1:
		x_train, x_test, y_train, y_test = train_test_split(features, np.ravel(labels), test_size=0.33, random_state=42) 	# separate train and test sets
	
		if classifier == 'RF':
			clf = RandomForestClassifier(n_estimators=400,n_jobs=4)		# create random forest
			clf.fit(x_train,y_train) 							# train classifier
			predictions = clf.predict_proba(np.ravel(x_test))			# generate predictions
			fpr,tpr,tr	= roc_curve(y_test,predictions,1)	
			score 	= roc_auc_score(y_test,predictions)
		if classifier == 'RR':
			clf = linear_model.Ridge(alpha = 0.5)					# create Ridge regression
			clf.fit(x_train,y_train) 							# train classifier
			predictions = clf.predict(x_test)						# generate predictions
			fpr,tpr,tr	= roc_curve(y_test,predictions,1)	
			score 	= roc_auc_score(y_test,predictions)	
		if classifier == 'GBR':	
			clf = GradientBoostingRegressor(n_estimators=100, loss='ls').fit(x_train, y_train)			
			clf
			
		plt.title(score)
		plt.plot(fpr,tpr)
	else:
		if classifier == 'RF':		
			clf = RandomForestClassifier(n_estimators=400,n_jobs=4)		# create random forest
		if classifier == 'RR':
			clf = linear_model.Ridge(alpha = 0.5)	
			
		clf.fit(features,labels) 							# train forest
		predictions = clf.predict(testFeatures)					# generate predictions
	

	makeSubmission(loc_submission,predictions,testids)
	
	
	
	
	
	