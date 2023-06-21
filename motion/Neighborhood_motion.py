# Module to construct the K closest neighbors (eigenvectors) of each amino_acids (local vectors included)
import numpy as np
from collections import defaultdict as dd
def Neighborhood_motion(file,indexes,m):
    """Compute Neighborhood regarding indexes of the K closest vectors
       indexes is a dictionnary of list containing labels of neighbors of the key-labeled amino_acid
       m is the number of motions """

    # to start with, we compute each amino_acids and their bound vectors
    f = open(file,'r')
    line = f.readline()
    amino_acids = dd(list)
    while line != "":
        line = line.replace("\n","")
        line = line.split("  ")
        i = 0
        while line[i] == '':
            i += 1
        index = int(line[i]) # index of the current amino_acid
        i += 3 # jump the center of mass
        for j in range(m):
            amino_acids[index].append(list(map(float,[line[k] for k in range(i+1+6*j,i+7+6*j)])))
        line = f.readline()
    # now, we just create the graph matrix
    G = []
    for index in amino_acids.keys(): # amino_acid per amino_acid
        neighborhood = []
        for label in indexes[index]: # list of indexes of the K-closest neighbors (itself included)
            neighborhood.append(amino_acids[label])
        G.append(neighborhood)
    return G

# tests
dic = dd(list)
for i in range(102):
    dic[i] = list(range(16))

Neighborhood_motion("pdb1b9e_rtb.txt",dic,10)
