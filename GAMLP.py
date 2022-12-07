import numpy as np
import pandas as pd
import DataSpilt as ds
from tqdm import trange # used for CV progress bar, simply replace trange with range if issues

# Dataset object: takes a string the name of the data set with or without .data at the end
#
# Dataset initializes the information related to each specific data set including the weights for each net work type
# Each dataset has 3 networks, 0 hidden layers, 1 hidden layer, and 2 hidden layers
# These are loaded upon creation of a network using saved weights from MLP

class Dataset:
    def __init__(self, data):
        if ".data" in data:
            self.name = data[:-5]  # sets data set name if input has .data
        else:
            self.name = data # sets name if input without .data

        self.path = "Data/" + self.name + ".data"  # saves path to the data file

        self.data = pd.read_csv(self.path, index_col=0, header=0)  # gets the data in a data frame
        self.samples = ds.getSamples(self.data)  # gets split data
        self.train = pd.concat(self.samples[:9])  # creates training data
        self.test = self.samples[9]  # creates test data

        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True
            self.classes = self.data['class'].unique() # gets classes
            self.output = len(self.classes)
            if self.output == 2:
                self.output = 1  # binary classification
        else:
            self.classification = False
            self.output = 1  # if regression one single output

        if self.output > 2:  # multi classification
            self.outputF = softMax
        elif self.classification:  # binary classification
            self.outputF = np.tanh
        else:  # regression
            self.outputF = identity

        self.networks = {i: Network(self, i) for i in [0, 1, 2]}  # generates the initial networks for each number of hidden layers

    # advances to the next training fold
    def nextFold(self):
        self.samples.append(self.samples.pop(0))
        self.train = pd.concat(self.samples[0:9])
        self.test = self.samples[9]


# Network object: takes a dataset and how many hidden layers should be in the network
#
# A network is a set of 3d weights initialized from the final weights used for each fold of CV in back propagation
# weights are like: [[10xNxM]_0, ... [10xNxM]_n]] where n is the number of hidden layers
# this optimizes the runtime of the program and eliminates for loops for each network
# when testing only the best of 10 networks is used
# each network has the methods for each of the evolutionary algorithms
# when running an algorithm it by default will perform cross validation

class Network:
    def __init__(self, dataset, layers):
        self.dataset = dataset  # sets what dataset the network belongs to
        w, bw = np.load('Weights/' + self.dataset.name + "/" + str(layers) + "/" + 'weights.npz'), np.load(
            'Weights/' + self.dataset.name + "/" + str(layers) + "/" + 'bweights.npz')
        # loads the weight matricies from MLP to the network for weights and bias weights
        self.weights = [w[key] for key in w]  # unpacks 3d weights, one for each fold in CV of mlp
        self.bweights = [np.swapaxes(bw[key], 1, 0) for key in bw]
        self.tWeight = self.weights # sets initial testing weights, updated later to 2d weights
        self.tBWeight = self.bweights
        self.layers = layers
        self.weightsCopy = self.weights.copy()  # creates copy to reset after each algorithm
        self.bweightsCopy = self.bweights.copy()
        self.results = {}

    # pushes all training vectors through the population network in one pass
    def evaluate(self, weights=None, test=False):
        if test: # if on the test fold of CV, uses only one set of 2d weight matrix to test
            data = self.dataset.test.to_numpy()
            weights = self.tWeight
            bweights = self.tBWeight
        else:  # if on train it uses a set of 3d weight matrices
            data = self.dataset.train.to_numpy()
            if not weights:  # if no weights input to the function
                weights = self.weights
                bweights = self.bweights
            else:  # uses weights input to the function (used with seeing if child perfromance increased without updating the current weights)
                weights, bweights = weights

        solutions = data[:, -1]  # sets solutions
        a = data[:, :-1]  # drops solution
        for idx, i in enumerate(weights[:-1]):  # feeding forward
            if idx == 0:
                z = np.einsum('ij,kjl-> ikl', a, i) + bweights[idx]  # matrix multiply
            if idx == 1:
                z = np.einsum('ijk,jkl -> ijl', a, i) + bweights[idx]
            a = np.tanh(z)

        w = weights[-1]
        if len(weights) == 1:
            z = np.einsum('ij,kjl-> ikl', a, w) + bweights[-1]
        else:
            z = np.einsum('ijk,jkl -> ijl', a, w) + bweights[-1]

        y = self.dataset.outputF(z).squeeze()  # calls output function of dataset

        if self.dataset.output > 1:
            # multi-classification takes the index of the largest output of softmax
            if test: axis = 1
            else: axis = 2
            y = self.dataset.classes[np.argmax(y, axis=axis)]

        elif self.dataset.classification:
            # binary classification rounds to index either 0 or 1 of classes
            y = self.dataset.classes[y.round().astype(int)]
        # regression is just the output
        performances = self.performance(y, solutions, test)
        return performances

    def performance(self, prediction, actual, test=False):
        np.seterr(invalid="ignore")
        if self.dataset.classification:
            performance = self.F1(prediction, actual)
        elif not test:
            performance = np.sum((prediction - actual[:, np.newaxis]) ** 2, axis=0) / prediction.shape[0]
        elif test:
            performance = np.sum((np.atleast_2d(prediction) - np.atleast_2d(actual))[0] ** 2, axis=0) / \
                          prediction.shape[0]
        np.seterr(invalid="warn")
        return performance

    def F1(self, pred, actual):
        # 3d vectorized F1 score
        np.seterr(divide='ignore', invalid='ignore')
        actual = actual.astype(int)
        classes = self.dataset.classes
        timesactual = np.bincount(actual, minlength=np.max(classes) + 1)
        # equivalent to TP + FP

        if len(pred.shape) > 1:
            correct = np.where(pred == actual[:, None], pred, 0).T
            timesguessed = np.apply_along_axis(np.bincount, 0, pred, minlength=np.max(classes) + 1).T
            # equivalent to TP + FN
            TP = np.apply_along_axis(np.bincount, 1, correct, minlength=np.max(classes) + 1)
        else:
            correct = np.where(pred == actual, pred, 0).T
            timesguessed = np.bincount(pred, minlength=np.max(classes) + 1).T
            TP = np.bincount(correct, minlength=np.max(classes) + 1)

        P = TP/ (timesactual)
        R = TP/ (timesguessed)
        f1 = (2 * (R * P) / (P + R))

        if len(pred.shape) > 1:
            classes = np.count_nonzero(~np.isnan(f1), axis=1)
            F1 = np.sum(np.nan_to_num(f1), axis=1) / classes
        else:
            classes = np.count_nonzero(~np.isnan(f1))
            F1 = np.sum(np.nan_to_num(f1)) / classes

        np.seterr(divide='warn', invalid='warn')
        return F1

    # reset weights of network object after performing algorithm
    def reset(self):
        self.weights = self.weightsCopy
        self.bweights = self.bweightsCopy

    # picks best performing weights on the train fold to use on the test fold
    def pickWeights(self, fArray=None):
        if self.dataset.classification:
            index = np.where(np.max(fArray))
        else:
            index = np.where(np.min(fArray))

        self.tWeight = [i[index] for i in self.weights]
        self.tBWeight = [i[:, index][0] for i in self.bweights]

    # performs genetic algorithm
    def geneticAlgorithm(self, prob=.005, SD=.001):

        def select(fitnesses, x=5):
            # should select x number of pairs weighting selection odds by fitness
            pSelection = fitnesses ** 2 / np.sum(fitnesses ** 2)
            if not self.dataset.classification:
                pSelection = (1 - pSelection) / np.sum(1 - pSelection)
            pairs = [np.random.choice(np.where(fitnesses)[0], p=pSelection, replace=False, size=2) for i in range(x)]
            return pairs

        def crossover(pairs):

            # iterate through each pair in the selected set of pairs

            # parents is a weight list that is a tuple with the bweight of that layer
            # it is 3d but the 3rd dimension is only 2 one for each parent
            # parents[n][i][0] is the nth pair, the ith layer, and the weights
            # parents[0][0][1] is the first pair, the first layer, and the bweights

            parents = [[(i[pair, :, :], self.bweights[idx][:, pair, :]) for idx, i in enumerate(self.weights)] for pair
                       in pairs]

            # xover is a similar structure except the xover matrix is 3d where the 3rd dimension is the amount of pairs
            # xover[i][0][n] is the ith layer, the weights xover and the nth pair
            # xover[2][1][2] is the 3rd layer, the bweights xover and the 3rd pair

            xover = [(np.random.choice([0, 1], p=[0.5, 0.5], size=(len(pairs), i.shape[1], i.shape[2])).astype(bool),
                      np.random.choice([0, 1], p=[0.5, 0.5], size=(
                          self.bweights[idx].shape[0], len(pairs), self.bweights[idx].shape[2])).astype(bool)) for
                     idx, i in
                     enumerate(self.weights)]

            # crossing over and putting things back together

            cW1 = [[np.choose(i[0][pidx], p[idx][0]) for pidx, p in enumerate(parents)] for idx, i in enumerate(xover)]
            cW2 = [[np.choose(~i[0][pidx], p[idx][0]) for pidx, p in enumerate(parents)] for idx, i in enumerate(xover)]
            cBW1 = [[np.choose(i[1][:, pidx], p[idx][1][0]) for pidx, p in enumerate(parents)] for idx, i in
                    enumerate(xover)]
            cBW2 = [[np.choose(~i[1][:, pidx], p[idx][1][0]) for pidx, p in enumerate(parents)] for idx, i in
                    enumerate(xover)]
            newW = [np.vstack([np.array(cW1[idx]), np.array(cW2[idx])]) for idx, i in enumerate(self.weights)]
            newBW = [np.vstack([np.array(cBW1[idx]), np.array(cBW2[idx])]).reshape(1, len(cBW2[0]) * 2, i.shape[2]) for
                     idx, i in enumerate(self.bweights)]

            return newW, newBW

        # randomly alter some genes in a given solution's chromosome, with fixed probability
        def Mutate(children, mutationProb=prob, mutationSD=SD):
            mutationBinaries = [np.random.choice([0, 1], p=[1 - mutationProb, mutationProb], size=i.shape) for i in
                                children]
            mutationTerms = [np.random.normal(0, mutationSD, size=i.shape) for i in children]
            mutated = [i + mutationBinaries[idx] * mutationTerms[idx] for idx, i in enumerate(children)]
            return mutated

        # perfroms one run through of algorithm
        def run():
            fitness = self.evaluate()
            size = len(fitness)
            pairs = select(fitness, 10)
            newW, newBW = crossover(pairs)
            self.weights = Mutate(newW)
            self.bweights = Mutate(newBW)
            if self.dataset.classification:
                best = np.max(fitness)
            else:
                best = np.min(fitness)
            return best

        # trains the weights and picks the best to train on
        def train(x=500):
            performanceTrain = []
            performanceTrain.append(np.max(self.evaluate()))
            for i in range(x):
                performanceTrain.append(run())
                # print(performance[-1], gen)
            self.pickWeights(performanceTrain[-1])

        # perfroms the CV
        performance = []
        for i in trange(10):
            train()
            perf = self.evaluate(test=True)
            print("Fold %s: %s"%(i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

    # Performs DiffEvo cv
    def diffEvo(self, xoP=.22, sF=.04):

        # xoP = crossover probabiliy
        # sF = scale factor
        def mutate(fitness):
            choices = np.where(fitness)[0]
            indexes = np.array([np.random.choice(np.where(choices != i)[0], size=3, replace=False) for i in choices])
            trialWeights = [i[indexes[:, 0]] + sF * (i[indexes[:, 1]] - i[indexes[:, 2]]) for i in self.weights]
            trialBWeights = [i[:, indexes[:, 0]] + sF * (i[:, indexes[:, 1]] - i[:, indexes[:, 2]]) for i in
                             self.bweights]

            trialVectors = trialWeights, trialBWeights

            return trialVectors

        def crossover(tV):
            # tV = trial vectors
            tW = tV[0]
            tBW = tV[1]

            xover = [(
                np.random.choice([0, 1], p=[1 - xoP, xoP], size=(tW[idx].shape[0], i.shape[1], i.shape[2])).astype(
                    bool),
                np.random.choice([0, 1], p=[1 - xoP, xoP], size=(
                    self.bweights[idx].shape[0], tBW[idx].shape[1], self.bweights[idx].shape[2])).astype(bool)) for
                idx, i in
                enumerate(self.weights)]

            # crossing over and putting things back together
            Wpairs = [[np.array([tW[idx][pidx], ii]) for pidx, ii in enumerate(i)] for idx, i in
                      enumerate(self.weights)]
            BWpairs = [[np.array([tBW[idx][:, pidx], ii]) for pidx, ii in
                        enumerate(i.reshape(i.shape[1], i.shape[0], i.shape[2]))] for idx, i in
                       enumerate(self.bweights)]

            cW1 = [np.array([np.choose(i[0][pidx], p) for pidx, p in enumerate(Wpairs[idx])]) for idx, i in
                   enumerate(xover)]
            cBW1 = [
                np.array([np.choose(i[1][:, pidx], p) for pidx, p in enumerate(BWpairs[idx])]).reshape(i[1].shape[0],
                                                                                                       i[1].shape[1],
                                                                                                       i[1].shape[2])
                for idx, i in enumerate(xover)]
            children = cW1, cBW1

            return children

        # picks where children performance better than parents and replaces it
        def pick(children, pFit):
            cFit = self.evaluate(children)
            keep = np.where(cFit >= pFit)
            replace = np.where(cFit < pFit)
            perf = np.min([cFit, pFit])
            if self.dataset.classification:
                keep = np.where(cFit <= pFit)
                replace = np.where(cFit > pFit)
                perf = np.max([cFit, pFit])
                if not np.any(cFit > pFit):
                    return self.weights, self.bweights, pFit
            if not np.any(cFit < pFit) and not self.dataset.classification:
                return self.weights, self.bweights, pFit
            bestW = [np.vstack([i[keep], children[0][idx][replace]]) for idx, i in enumerate(self.weights)]
            bestBW = [np.vstack([i[:, keep][0][0], children[1][idx][:, replace][0][0]])[np.newaxis, :, :] for idx, i in
                      enumerate(self.bweights)]

            try:
                fitness = np.insert(pFit[keep], -1, cFit[replace])
            except:
                fitness = cFit[replace]

            return bestW, bestBW, fitness

            # mutates then crosses over weights

        # performs operations based on the input of a fitness
        def evolve(fitness):
            self.weights, self.bweights, fitness = pick(crossover(mutate(fitness)), fitness)
            return fitness

        # runs one pass through of algorithm
        def run():
            fitness = self.evaluate()
            next = evolve(fitness)
            if self.dataset.classification:
                best = np.max(next)
            else:
                best = np.min(next)
            return best

        # trains weights using diffEvo
        def train(x=500):
            performanceTrain = []
            performanceTrain.append(np.max(self.evaluate()))
            for i in range(x):
                performanceTrain.append(run())
            self.pickWeights(performanceTrain[-1])

        # performs CV
        performance = []
        for i in trange(10):
            train()
            perf = self.evaluate(test=True)
            print("Fold %s: %s"%(i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

    # performing CV on PSO
    def PSO(self, inertia=.5, c1=1.486, c2=1.486):
        c = 2.05  # to do use velocity constriction c+c > 4
        v = [np.zeros(shape=i.shape) for i in self.weights]
        bv = [np.zeros(shape=i.shape) for i in self.bweights] # intial velocities 0
        # initialize pBest positions and fitnesses, gBest position and gBest fitness
        bestF = self.evaluate()  # initial fitness is best fitness
        bestW = self.weights  # initial weights are best weights
        bestBW = self.bweights
        gB = np.where(np.min(bestF) == bestF)[0][0]  # where the current global best model is
        gBF = bestF[gB]  # the current global best fitness

        # update the personal bests for all solutions
        def updatePBests(fitness):

            # finds where performance was better than personal best
            if self.dataset.classification:
                updates = np.where(fitness > bestF)
            else:
                updates = np.where(fitness < bestF)
            # updates pbest fitness
            bestF[updates] = fitness[updates]

            # replaces the weights that improved with the new pbest
            for idx, i in enumerate(self.weights):
                bestW[idx][updates] = i[updates]

            for idx, i in enumerate(self.bweights):
                bestBW[idx][:,updates] = i[:,updates]

        #update the global best fitness
        def updateGBest(fitness):

            # finds where, if any the current fitness is greater than the global best
            if self.dataset.classification:
                update = np.where(np.max(fitness) > gBF)
            else:
                update = np.where(fitness < gBF)

            # if none returns prev global best
            if len(update[0]) == 0:
                return gB, gBF

            # if the fitness did improve sets new global best
            if self.dataset.classification:
                newGB = np.max(fitness)
            else:
                newGB = np.min(fitness)

            # sets global best index
            update = np.where(fitness == newGB)[0][0]

            return update, newGB

        # update the velocities of each particle
        def updateVelocities():

            # constriction factor
            phi = c*2
            k = 2 / (abs(2 - phi - np.sqrt(phi**2-4*phi)))

            r1, r2 = np.random.uniform(0, 1, size=2)

            # sets new velocities using velocity constriction
            NewV = [k*(v[idx] + c * r1 *(bestW[idx] - i) + c * r2 *(bestW[idx][gB] - i)) for idx, i in enumerate(self.weights)]
            NewBV = [k*(bv[idx] + c * r1 *(bestBW[idx] - i) + c * r2 *(bestBW[idx][:,gB] -i)) for idx, i in enumerate(self.bweights)]
            return NewV, NewBV


        # update the positions of each particle
        def updatePositions(v, bv):

            # adds velocities to prev position
            self.weights = [i+v[idx] for idx, i in enumerate(self.weights)]
            self.bweights = [i+bv[idx] for idx, i in enumerate(self.bweights)]

            return

        # one run through the PSO algo
        def run():
            fitness = self.evaluate()
            updatePBests(fitness)
            globalBest, globalBestFitness = updateGBest(fitness)
            newV, newBv = updateVelocities()
            updatePositions(newV, newBv)
            return globalBest, globalBestFitness, newV, newBv

        # trains a set of weights using PSO
        def train(x = 500):
            nonlocal gB, gBF, v, bv
            gB, gBF = updateGBest(bestF)
            performanceTrain = []
            performanceTrain.append(gBF)
            for i in range(x):
                gB, gBF, v, bv = run()
            self.pickWeights(bestF)

        # performs the CV for PSO
        performance = []
        for i in trange(10):
            train()
            perf = self.evaluate(test=True)
            #print("Fold %s: %s" % (i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

# used for regression outputs
def identity(x):
    return x

# used for multi-classification outputs
def softMax(o):
    val = np.exp(o) / np.sum(np.exp(o))
    return val


glass = Dataset("glass")
print(glass.networks[2].PSO())
# this runs the cross validation
