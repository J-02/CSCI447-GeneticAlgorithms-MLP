import numpy as np
import pandas as pd
from MLP import MLP
import DataSpilt as ds
import random
from line_profiler_pycharm import profile
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import trange


class Dataset:
    def __init__(self, data):
        if ".data" in data:
            self.name = data[:-5]
        else:
            self.name = data

        self.path = "Data/" + self.name + ".data"

        self.data = pd.read_csv(self.path, index_col=0, header=0)
        self.samples = ds.getSamples(self.data)  # gets split data
        self.train = pd.concat(self.samples[:9])  # creates training data
        self.test = self.samples[9]  # creates test data

        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True
            self.classes = self.data['class'].unique()
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

        self.networks = {i: Network(self, i) for i in [0, 1, 2]}

    def nextFold(self):
        self.samples.append(self.samples.pop(0))  # rotation of folds
        self.train = pd.concat(self.samples[0:9])
        self.test = self.samples[9]


class Network:
    def __init__(self, dataset, layers):
        self.dataset = dataset
        w, bw = np.load('Weights/' + self.dataset.name + "/" + str(layers) + "/" + 'weights.npz'), np.load(
            'Weights/' + self.dataset.name + "/" + str(layers) + "/" + 'bweights.npz')
        self.weights = [w[key] for key in w]
        self.bweights = [np.swapaxes(bw[key], 1, 0) for key in bw]
        self.tWeight = self.weights
        self.tBWeight = self.bweights
        self.layers = layers
        self.weightsCopy = self.weights.copy()
        self.bweightsCopy = self.bweights.copy()
        self.results = {}

    # pushes all training vectors through the population network in one pass
    @profile
    def evaluate(self, weights=None, test=False):
        if test:
            data = self.dataset.test.to_numpy()
            weights = self.tWeight
            bweights = self.tBWeight
        else:
            data = self.dataset.train.to_numpy()
            if not weights:
                weights = self.weights
                bweights = self.bweights
            else:
                weights, bweights = weights

        solutions = data[:, -1]
        a = data[:, :-1]
        for idx, i in enumerate(weights[:-1]):
            if idx == 0:
                z = np.einsum('ij,kjl-> ikl', a, i) + bweights[idx]
            if idx == 1:
                z = np.einsum('ijk,jkl -> ijl', a, i) + bweights[idx]
            a = np.tanh(z)

        w = weights[-1]
        if len(weights) == 1:
            z = np.einsum('ij,kjl-> ikl', a, w) + bweights[-1]
        else:
            z = np.einsum('ijk,jkl -> ijl', a, w) + bweights[-1]

        y = self.dataset.outputF(z).squeeze()
        performances = self.performance(y, solutions, test)
        return performances

    def performance(self, prediction, actual, test=False):
        np.seterr(invalid="ignore")
        if self.dataset.classification:
            performance = f1_score(prediction, actual)
        elif not test:
            performance = np.sum((prediction - actual[:, np.newaxis]) ** 2, axis=0) / prediction.shape[0]
        elif test:
            performance = np.sum((np.atleast_2d(prediction)- np.atleast_2d(actual))[0] ** 2, axis=0) / prediction.shape[0]
        np.seterr(invalid="warn")
        return performance

    # reset weights of network object after performing algorithm
    def reset(self):
        self.weights = self.weightsCopy
        self.bweights = self.bweightsCopy

    # picks best performing weights on the train fold to use on the test fold
    def pickWeights(self, fArray):
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

        @profile
        def run():
            fitness = self.evaluate()
            size = len(fitness)
            pairs = select(fitness, 10)
            newW, newBW = crossover(pairs)
            self.weights = Mutate(newW)
            self.bweights = Mutate(newBW)
            mF = np.max(fitness)
            #print(mF)
            return mF

        def train(x=500):
            performanceTrain = []
            performanceTrain.append(np.max(self.evaluate()))
            for i in trange(x):
                performanceTrain.append(run())
                # print(performance[-1], gen)
            self.pickWeights(performanceTrain[-1])
            print(performanceTrain[-1])

        performance = []
        for i in range(10):
            train()
            perf = self.evaluate(test=True)
            print("Fold %s: %s"%(i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

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

        def pick(children, pFit):
            cFit = self.evaluate(children)
            keep = np.where(cFit >= pFit)
            replace = np.where(cFit < pFit)
            perf = np.min([cFit, pFit])
            if self.dataset.classification:
                keep, replace = replace, keep
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

        def evolve(fitness):
            self.weights, self.bweights, fitness = pick(crossover(mutate(fitness)), fitness)
            return fitness

        def run():
            fitness = self.evaluate()
            next = evolve(fitness)
            return np.max(next)

        def train(x=500):
            performanceTrain = []
            performanceTrain.append(np.max(self.evaluate()))
            for i in trange(x):
                performanceTrain.append(run())
            self.pickWeights(performanceTrain[-1])
            print(performanceTrain[-1])

        performance = []
        for i in range(10):
            train()
            perf = self.evaluate(test=True)
            print("Fold %s: %s"%(i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

    def PSO(self, inertia=.5, c1=1.486, c2=1.486):
        # initialize pBest positions and fitnesses, gBest position and gBest fitness
        pBestPos = self.weights.copy()
        pBests = self.evaluate()
        gBestPos = [i[np.argmin(pBests)] for i in self.weights]
        self.gBest = np.argmin(pBests)

        # initialize velocities randomly, might want to tune bounds later
        velocities = []
        for i in range(len(pBests)):
            velocities.append([np.random.uniform(-.1,.1, size=i[0].shape) for i in self.weights])

        # update the personal bests for all solutions
        def updatePBests(fitness):
            # for each particle
            for i in range(len(fitness)):

                # if new fitness is the best the particle's seen so far
                if fitness[i] < pBests[i]:

                    # update the personal best fitness of the particle
                    pBests[i] = fitness[i]

                    # as well as the position that this fitness was found at
                    [pBestPos[0][i], pBestPos[1][i], pBestPos[2][i]] = [j[i] for j in self.weights]


        #update the global best fitness
        def updateGBest(fitness):
            # if any of the newly found fitnesses are better than the current global best
            if(np.argmin(fitness) < self.gBest):

                # update the position that this global best was found at
                gBestPos = [i[np.argmin(fitness)] for i in self.weights]

                # as well as storing what this global best fitness is
                self.gBest = fitness[np.argmin(fitness)]

        # update the velocities of each particle
        def updateVelocities():
            # for each particle
            for i in range(len(pBests)):

                # generate two random numbers ~ U[0,1]
                r1, r2 = np.random.uniform(0, 1, size=2)

                # compute the inertia component of the velocity equation
                inertia_component = [inertia*j for j in velocities[i]]

                # the cognitive component (taking personal best into account)
                cognitive_component = [c1*r1*(j[i]-k[i]) for j, k in zip(pBestPos, self.weights)]

                # and the social component (taking global best into account)
                social_component = [c2*r2*(j-k[i]) for j, k in zip(gBestPos, self.weights)]

                # combine all three terms for the velocity of each particle
                velocities[i] = [np.add(j,np.add(k,l)) for j, k, l in zip(inertia_component,cognitive_component,social_component)]

        # update the positions of each particle
        def updatePositions():

            # for each particle
            for i in range(len(pBests)):

                # add the velocity of each particle onto the current position
                [self.weights[0][i], self.weights[1][i], self.weights[2][i]] = [j[i] + k for j, k in zip(self.weights, velocities[i])]

            return self.weights

        # runner for PSO
        def run():
            updateVelocities()
            updatePositions()
            fitness = self.evaluate()
            updatePBests(fitness)
            updateGBest(fitness)
            fitness = self.evaluate()
            return np.min(fitness)


        while(True):
            performance = run()
            print(performance)




def identity(x):
    return x


def softMax(o):
    val = np.exp(o) / np.sum(np.exp(o))
    return val


def f1_score(prediction, actual):
    np.seterr(divide='ignore', invalid='ignore')
    m = pd.crosstab(prediction, actual)  # confusion matrix
    i = m.index.union(m.columns)
    m = m.reindex(index=i, columns=i, fill_value=0)  # makes the matrix square
    m.fillna(0)  # fills na values w/ 0
    P = precision(m)  # calculates precision
    R = recall(m)  # calcaultes recall
    f1 = 2 * (R * P) / (P + R)  # calculates f1
    f1 = f1[~np.isnan(f1)]  # drops na values
    f1 = np.sum(f1) / f1.shape[0]  # gets average of all f1 scores
    np.seterr(divide='warn', invalid='warn')
    return f1


def precision(m):
    M = m.to_numpy()
    diag = np.diag(M)  # true positives
    p = diag / np.sum(M, axis=0)  # true positives / TP + false positives
    return p


def recall(m):
    M = m.to_numpy()
    diag = np.diag(M)  # true positives
    r = diag / np.sum(M, axis=1)  # true positives / TP + false negatives
    return r


abalone = Dataset("abalone")
print(abalone.networks[2].geneticAlgorithm())
# this runs cross validation
