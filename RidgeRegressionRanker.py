import numpy as np
import math as math

basepath = 'OHSUMED\\QueryLevelNorm\\'
folders = ['Fold1', 'Fold2','Fold3','Fold4','Fold5']
trainfile = 'train.txt'
validfile = 'vali.txt'
testfile = 'test.txt'

regularizationvec = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10]
kvec = [1, 3, 5]
NUMFEATURES = 45

def GetFeatures(line):
  featurestr = line[line.find(' ',2):line.find('#')].strip()
  features = np.zeros(NUMFEATURES)
  featureindex = 0
  for s in featurestr.split(): 
    features[featureindex] = float(s.split(':')[1])
    featureindex += 1
  return features
  
def LoadDataset(file, linecount):
  f = open(file, 'rU')
  
  X  = np.zeros((linecount, NUMFEATURES+1))
  QueryIds = np.zeros(linecount, dtype=int)
  X[:,0] = 1
  Y = np.zeros(linecount, dtype = int)
  lineindex = 0
  for line in f: 
    #NOTE: I think there might be a bug here: While doing regression, we should regress on the gain(label) instead of the actual label
    #I checked up SVMRank, and they are quite contented with counting the number of mislabeled pairs. They do not bother with what was the weight of a misclassification
    Y[lineindex] = int(line[0]) 
    qidindex = line.find('qid:')
    QueryIds[lineindex] = int(line[qidindex + 4: line.find(' ',qidindex)])
    X[lineindex,1:] = GetFeatures(line)
    lineindex += 1
  
  f.close()
  return (X, Y, QueryIds)

def DoRidgeRegression(Xtrain, Ytrain, regularization):
  betas = np.zeros(NUMFEATURES)
  Xtrainmat = np.matrix(Xtrain)
  mat0 = np.eye(NUMFEATURES+1)
  mat0[0,0] = 0 #no regularization for the offset term of the parameter vector
  mat1 = Xtrainmat.T * Xtrainmat + regularization * mat0
  mat1inv = np.linalg.inv(mat1)
  mat2 = np.dot(Xtrainmat.T, Ytrain).T
  betas = np.dot(mat1inv, mat2)
  
  retbetas = np.zeros(betas.shape[0])
  for i in range(betas.shape[0]):retbetas[i] = betas[i,0]
  return retbetas

def GetRankedLabels(Yrankscores, Y, QueryIds):
  dict = {}
  for i in range(len(QueryIds)):
    if QueryIds[i] in dict: dict[QueryIds[i]].append( (Yrankscores[i], Y[i]) )
    else:
      vec = []
      vec.append( (Yrankscores[i], Y[i]) )
      dict[QueryIds[i]] = vec
  
  retdict = {}
  for queryid in dict.keys():
    retdict[queryid] = [tuple[1] for tuple in sorted(dict[queryid], reverse = True)]
  return retdict
    
def GetPrecisionatK(eval, k):
  precisionatk = 0.0
  for queryid in eval:
    precisionatk += float(np.sum(np.array(eval[queryid][:k], dtype=int) == 2))/float(k)
  precisionatk /= float(len(eval))
  return precisionatk
    
def GetMAP(eval):
  map = 0.0
  for queryid in eval:
    denominator = np.sum(np.array(eval[queryid], dtype=int) == 2)
    if denominator == 0: denominator = 1
    val = 0.0
    for pos in range(len(eval[queryid])):
      if eval[queryid][pos] == 2:
        val += float(np.sum(np.array(eval[queryid][:pos+1], dtype=int) == 2)) / float(pos+1)
    val /= denominator
    map += val
  map /= float(len(eval))
  return map
  
def GetNDCGatK(eval, k):
  ndcgatk = 0.0
  discounts = np.zeros(k)
  for i in range(k): discounts[i] = float(1.0) / math.log(i+2, 2)
  
  for queryid in eval:
    gain = np.array([2**val-1 for val in eval[queryid][:k]])
    idealgain = np.array([2**val-1 for val in sorted(eval[queryid][:k], reverse = True)])
    denominator = np.dot(idealgain, discounts)
    if denominator == 0.0: denominator = 1.0
    ndcgatk += float(np.dot(gain, discounts)) / float(denominator)
  ndcgatk /= float(len(eval))
  return ndcgatk

def main():
  trainresults = {}
  validresults = {}
  testresults = {}
  
  for regularization in regularizationvec:
    trainresultvec = np.zeros(3+1+3)
    validresultvec = np.zeros(3+1+3)
    testresultvec = np.zeros(3+1+3)
    
    for folder in folders:
      trainlinecount = sum(1 for tmpline in open(basepath + folder + '\\' + trainfile, 'rU'))
      validlinecount = sum(1 for tmpline in open(basepath + folder + '\\' + validfile, 'rU'))
      testlinecount = sum(1 for tmpline in open(basepath + folder + '\\' + testfile, 'rU'))
      
      Xtrain, Ytrain, trainQueryIds = LoadDataset(basepath + folder + '\\' + trainfile, trainlinecount) 
      Xvalid, Yvalid, validQueryIds = LoadDataset(basepath + folder + '\\' + validfile, validlinecount)
      Xtest, Ytest, testQueryIds = LoadDataset(basepath + folder + '\\' + testfile, testlinecount)
      
      betas = DoRidgeRegression(Xtrain, Ytrain, regularization)
      Ytrainrankscores = np.dot(Xtrain, betas)      
      Yvalidrankscores = np.dot(Xvalid, betas)
      Ytestrankscores = np.dot(Xtest, betas)
      
      traineval = GetRankedLabels(Ytrainrankscores, Ytrain, trainQueryIds)
      valideval = GetRankedLabels(Yvalidrankscores, Yvalid, validQueryIds)
      testeval = GetRankedLabels(Ytestrankscores, Ytest, testQueryIds)
      
      for i in range(len(kvec)):
        trainresultvec[i] += GetPrecisionatK(traineval, kvec[i])
        validresultvec[i] += GetPrecisionatK(valideval, kvec[i])
        testresultvec[i] += GetPrecisionatK(testeval, kvec[i])
      
      trainresultvec[3] += GetMAP(traineval)
      validresultvec[3] += GetMAP(valideval)
      testresultvec[3] += GetMAP(testeval)
      
      for i in range(len(kvec)):
        trainresultvec[4+i] += GetNDCGatK(traineval, kvec[i])
        validresultvec[4+i] += GetNDCGatK(valideval, kvec[i])
        testresultvec[4+i] += GetNDCGatK(testeval, kvec[i])
      
    trainresultvec /= float(len(folders))
    validresultvec /= float(len(folders))
    testresultvec /= float(len(folders))
    
    trainresults[regularization] = trainresultvec
    validresults[regularization] = validresultvec
    testresults[regularization] = testresultvec
    
    print regularization, trainresultvec
    print regularization, validresultvec
    print regularization, testresultvec
if __name__ == '__main__':
  main()