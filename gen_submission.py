"Generate a Kaggle submission file from VW predictions"

loc_preds = 'C:/Users/tushar.tank/Documents/GitHub/3PiShopper/Data/treePreds.txt'
loc_submission = 'C:/Users/tushar.tank/Documents/GitHub/3PiShopper/Data/kaggle.submission.txt'
loc_test = 'C:/Users/tushar.tank/Documents/GitHub/3PiShopper/Data/testHistory.csv'

def generate_submission(loc_preds, loc_test, loc_submission):
    preds = {}
    for e, line in enumerate( open(loc_preds) ):
        row = line.strip().split(" ")
        preds[ row[1] ] = row[0]
        
    
    with open(loc_submission, "wb") as outfile:
        for e, line in enumerate( open(loc_test) ):
            if e == 0:
                outfile.write( "id,repeatProbability\n" )
            else:
                row = line.strip().split(",")
                if row[0] not in preds:
                    outfile.write(row[0]+",.001\n")
                    print row[0]
                else:
                    outfile.write(row[0]+","+preds[row[0]]+"\n")
                    
#generate_submission(loc_preds, loc_test, loc_submission)

if __name__ == '__main__':
    generate_submission(loc_preds, loc_test, loc_submission)