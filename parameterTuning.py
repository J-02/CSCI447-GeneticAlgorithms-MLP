import numpy as np
from MLP import MLP
#  Parameter tuning: tuns the set of NN parameters from given vectors of possible values

def tuneParameters(file, nodes, step_size, momentum):
    #import dataset

    # parameter grid for grid search in 3 dimensions
    parameterHypercube = np.zeros([len(nodes), len(nodes), len(step_size), len(momentum)])

    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            for k in range(0, len(step_size)):
                for l in range(len(momentum)):
                    #fit MLP
                    print('Trying 2 layers with',nodes[i],'and',nodes[j],'nodes', step_size[k],'step size',momentum[l],'momentum')
                    combo_MLP = MLP(file, [nodes[i],nodes[j]], step_size[k], momentum[l])
                    parameterHypercube[i,j, k, l] = combo_MLP.Train()
                    print('F1: ',parameterHypercube[i, j, k, l])


    F1_max = 0
    a = 0
    b = 0
    c = 0
    d = 0
    #  iterates through entire hypercube and finds combo of parameters that return highest F1
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            for k in range(0, len(step_size)):
                for l in range(len(momentum)):
                    if parameterHypercube[i, j, k, l] > F1_max:
                        a = i
                        b = j
                        c = k
                        d = l
                        F1_max = parameterHypercube[i, j, k, l]


    print('Best Values\n',nodes[a],'and',nodes[b], 'nodes,', step_size[c], 'step size,', momentum[d], 'momentum:')
    print('F1:',F1_max)

tuneParameters('breast-cancer-wisconsin.data', [1,3,5,7,9], [.01, .05, .1, .2, .4, .5, .6, .8, 1], [0])