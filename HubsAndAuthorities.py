import numpy as np

N = 6
G = {}
G[0] = []
G[1] = []
G[2] = [0]
G[3] = [0]
G[4] = [0, 1]
G[5] = [1]

N += 3
G[6] = []
G[7] = []
G[8] = []

MAXITERATIONS = 20
EPSILON = 1e-9

for vec1 in [[6], [6,0], [6,1], [6,0,1]]:
  G[7] = vec1
  for vec2 in [[6], [6,0], [6,1], [6,0,1]]:
    G[8] = vec2

    hubs = np.ones(N)
    auths = np.ones(N)

    print 'choice for Y and Z'
    print G[7]
    print G[8]
    for iteration in range(0, MAXITERATIONS):
      if iteration < 2 or iteration == MAXITERATIONS-1: print 'iteration:' + str(iteration)
      for page in range(0, N): 
        auths[page] = 0
        for page2 in range(0, N):
          if page in G[page2]:
            auths[page] += hubs[page2]
      norm = np.sum(auths)#np.sqrt(np.dot(auths, auths))
      if iteration < 2 or iteration == MAXITERATIONS-1: print 'authorities'
      if iteration < 2 or iteration == MAXITERATIONS-1: print auths
      for page in range(0, N): auths[page] /= norm
      if iteration < 2 or iteration == MAXITERATIONS-1: print auths
      
      for page in range(0, N):
        hubs[page] = 0
        for page2 in G[page]: hubs[page] += auths[page2]
      norm = np.sum(hubs)#np.sqrt(np.dot(hubs, hubs))
      if iteration < 2 or iteration == MAXITERATIONS-1: print 'hubs'
      if iteration < 2 or iteration == MAXITERATIONS-1: print hubs
      for page in range(0, N): hubs[page] /= norm
      if iteration < 2 or iteration == MAXITERATIONS-1: print hubs
  
    if auths[6] > auths[1] + auths[1]*EPSILON: print 'X second largest'