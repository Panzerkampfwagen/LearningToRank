
invindex = {}

def GetMatchingDocuments(query):
  print 'Get Matching Documents for query:'+query
  resdict = {}
  for term in query.split():
    if len(resdict) == 0: 
      resdict = invindex[term].copy()
      continue
    tmpdict = {}
    for doc in invindex[term]:
      if doc in resdict:
        for pos in resdict[doc]:
          if (pos+1) in invindex[term][doc]:
            if doc in tmpdict: tmpdict[doc].append(pos+1)
            else: 
              tmpvec = []
              tmpvec.append(pos+1)
              tmpdict[doc] = tmpvec
    resdict = tmpdict.copy()
  
  for doc in resdict:
    vec = []
    for pos in resdict[doc]: vec.append(pos-len(query.split()) + 1)
    resdict[doc] = vec
    print doc, resdict[doc]
  
  return resdict.keys()
    
def BuildInvertedIndex():
  f = open('input.txt', 'rU')
  for s in f:
    dict = {}
    for doctermlist in s[(s.find(':')+2):].split(';'):
      tmps = doctermlist.strip().replace(' ', '').split(':')
      if tmps[0] == '': continue
      key = int(tmps[0])
      vec = []
      for element in tmps[1][1:-1].split(','):
        if element == '': continue
        vec.append(int(element))
      dict[key] = vec
    invindex[s[:s.find(':')]] = dict

  for key in invindex.keys(): print key, invindex[key]

def main():
  BuildInvertedIndex()
  query1 = 'fools rush in'
  query2 = 'angels fear to tread'
  
  matchingDocs1 = GetMatchingDocuments(query1)
  matchingDocs2 = GetMatchingDocuments(query2)
  
  print matchingDocs1
  print matchingDocs2
  
  retvec = []
  for doc in matchingDocs1:
    if doc in matchingDocs2:
      retvec.append(doc)
      
  print retvec

if __name__ == '__main__':
  main()