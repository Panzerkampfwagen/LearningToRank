import numpy as np
import math as math

basepath = 'D:\\gatech\\cse6240\\homeworks\\hw1\\OHSUMED\\QueryLevelNorm\\'
folders = ['Fold1', 'Fold2','Fold3','Fold4','Fold5']
rankscorefiles = ['rankscore1e-4', 'rankscore1e-3', 'rankscore1e-2', 'rankscore1e-1', 'rankscore2e-1', 'rankscore5e-1', 'rankscore1', 'rankscore2', 'rankscore5', 'rankscore10']
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
    Y[lineindex] = int(line[0])
    qidindex = line.find('qid:')
    QueryIds[lineindex] = int(line[qidindex + 4: line.find(' ',qidindex)])
    X[lineindex,1:] = GetFeatures(line)
    lineindex += 1
  
  f.close()
  return (X, Y, QueryIds)

  
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
  
  for regularizationindex in range(len(regularizationvec)):
    trainresultvec = np.zeros(3+1+3)
    validresultvec = np.zeros(3+1+3)
    testresultvec = np.zeros(3+1+3)
    
    for folder in folders:
      trainlinecount = sum(1 for tmpline in open(basepath + folder + '\\' + 'train.txt', 'rU'))
      validlinecount = sum(1 for tmpline in open(basepath + folder + '\\' + 'vali.txt', 'rU'))
      testlinecount = sum(1 for tmpline in open(basepath + folder + '\\' + 'test.txt', 'rU'))
      
      Xtrain, Ytrain, trainQueryIds = LoadDataset(basepath + folder + '\\' + 'train.txt', trainlinecount) 
      Xvalid, Yvalid, validQueryIds = LoadDataset(basepath + folder + '\\' + 'vali.txt', validlinecount)
      Xtest, Ytest, testQueryIds = LoadDataset(basepath + folder + '\\' + 'test.txt', testlinecount)
      
      Ytrainrankscores = np.loadtxt(basepath + folder + '\\train' + rankscorefiles[regularizationindex] + '.txt')
      Yvalidrankscores = np.loadtxt(basepath + folder + '\\valid' + rankscorefiles[regularizationindex] + '.txt')
      Ytestrankscores = np.loadtxt(basepath + folder + '\\test' + rankscorefiles[regularizationindex] + '.txt')
      
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
    
    trainresults[regularizationvec[regularizationindex]] = trainresultvec
    validresults[regularizationvec[regularizationindex]] = validresultvec
    testresults[regularizationvec[regularizationindex]] = testresultvec
    
    print regularizationvec[regularizationindex], trainresultvec
    print regularizationvec[regularizationindex], validresultvec
    print regularizationvec[regularizationindex], testresultvec
if __name__ == '__main__':
  main()