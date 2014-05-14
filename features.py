# -*- coding: utf-8 -*-
"""
Created on Tue May 13 11:52:33 2014

Correlate product brands with repeat purchase

@author: miguel.rufino
"""


def makeSmallerTransactionsFile(loc_trans,newLoc):
	
	with open(newLoc, 'wb') as outfile:
		for e, line in enumerate(open(loc_trans)):
			outfile.write(line)	
			if e==50000:
				break;

def makeCartIndicesFile(loc_trans,loc_carts):
	inCart = 0	# new cart flag
	cartCount = 0;
	# open cart file to write indices to
	with open(loc_carts, 'wb') as outfile:
		for e, line in enumerate(open(loc_trans)):	# for each line in transactions figure out what cart it belongs to
			if e==1:
				currentCartID = [line.split(',')[0], line.split(',')[0], line.split(',')[6]]   # init cartID 			
			if e>1: # skip header
				if inCart!=1:												# if current transaction is in a new cart
					startOfCart = e-1;										# record start of cart	
					inCart = 1;												# flag for same cart
					
				lastCartID = currentCartID										# store user id, chain and date as the cartID
				currentCartID = [line.split(',')[0], line.split(',')[1], line.split(',')[6]]	
				
				if currentCartID != lastCartID:										# if ID changes it's a new cart
					outfile.write(str(cartCount) +',' + str(startOfCart)+',' + str(e-1) + '\n')	# write cart indices to file 'cartID,startIndex,stopIndex'
					cartCount = cartCount + 1										# count carts
					inCart=0;													# new cart flag
			
			if e % 100000 == 0:	# print count at every 50000
				print e

			if e==50000:		# only process 50000	
				break;


def reduce_data_carts(loc_offers, loc_trans, loc_reduced, loc_carts):
	
	# make dictionary of offers
	# find if any item in cart contains that offer
	# if so write entire cart to loc_reduced	
	
	reduced = 0;	
	
	#get all categories and comps on offer in a dict
	offers_cat = {}
	offers_co  = {}
	offers_dept = {}
	for e, line in enumerate( open(loc_offers) ):
		offers_cat[ line.split(",")[1] ] 	= 1
		offers_co[ line.split(",")[3] ] 	= 1
		offers_dept[ line.split(",")[6] ] 	= 1		
	#open output file
	with open(loc_reduced, "wb") as outfile, open(loc_trans) as trans:			# open file to write to and transactions file
		outfile.write(trans.readline()) 								# write off header
		for c, carts in enumerate( open( loc_carts ) ):					# for each cart
			if c>0:	

				buffer = []
				for i in range( int(carts.split(',')[1]) , int(carts.split(',')[2]) ):		# read cart transactions to buffer
					buffer.append(trans.readline())
					
				# find offer in buffer
				for items in buffer:
					# if found write buffer to file
					if items.split(",")[3] in offers_cat or items.split(",")[4] in offers_co or items.split(",")[2] in offers_dept: 		
						for line in buffer:					
							outfile.write(line)		
							reduced += 1
						break;											
		
			#progress
			if c % 10000 == 0:
				print c,reduced



if __name__ == '__main__':

	loc_trans 	= 'transactions.csv'
	newLoc 	= 'smaller'+loc_trans
	loc_train 	= 'trainHistory.csv'
	loc_carts  	= 'cartIndices'
	loc_reduced	= 'reducedCarts'
	loc_offers	= 'offersDept.csv'
	
#	makeCartIndicesFile(loc_trans,loc_carts)	
	reduce_data_carts(loc_offers, loc_trans, loc_reduced, loc_carts)
	
	
#	makeSmallerTransactionsFile(loc_trans,newLoc)
#	productID = []
#	
#	# make dictionary of train data based on ids
#	train_ids={}
#	for e, line in enumerate( open(loc_train) ):
#		if e > 0:
#			row = line.strip().split(",")
#			train_ids[row[0]] = row	
#	
#	# list = [id,category,company,brand,repeattrips]
#	for e, line in enumerate(open(newLoc)):
#		if e>1:
#			if train_ids[line.split(',')[0]] in dict.keys():
#				productID.append([line.split(',')[0], line.split(',')[3], line.split(',')[4], line.split(',')[5], train_ids[line.split(',')[0]][4] ])
#		if e % 50000 == 0:
#			print e

	# for each unique product id what's the average purchase quantity?
