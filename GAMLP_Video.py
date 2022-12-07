import numpy as np
import pandas as pd
from MLP import MLP
import DataSpilt as ds
import random
from line_profiler_pycharm import profile
import matplotlib.pyplot as plt

def identity(x):
    return x


def softMax(o):
    val = np.exp(o) / np.sum(np.exp(o))
    return val


def f1_score(prediction, actual):
    np.seterr(divide='ignore', invalid='ignore')
    if len(prediction.shape) == 1:
        m = pd.crosstab(prediction, actual.astype(int))  # confusion matrix
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
    else:
        F1 = []
        for i in prediction.T:
            m = pd.crosstab(i, actual.astype(int))  # confusion matrix
            i = m.index.union(m.columns)
            m = m.reindex(index=i, columns=i, fill_value=0)  # makes the matrix square
            m.fillna(0)  # fills na values w/ 0
            P = precision(m)  # calculates precision
            R = recall(m)  # calcaultes recall
            f1 = 2 * (R * P) / (P + R)  # calculates f1
            f1 = f1[~np.isnan(f1)]  # drops na values
            f1 = np.sum(f1) / f1.shape[0]  # gets average of all f1 scores
            F1.append(f1)
        np.seterr(divide='warn', invalid='warn')
        return np.array(F1)

def CM(pred, actual):
    classes = np.unique()
    TP = np.sum((actual == pred), axis=1)
    TN = np.sum((actual == pred)[:,~actual], axis=1)
    FP = np.sum((actual != pred)[:,~actual], axis=1)
    FN = np.sum((actual != pred)[:,actual], axis=1)
    f1 = (2*TP)/(2*TP + FP + FN)
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
    def evaluate(self, weights=None, test=False, eval=False):
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

        if self.dataset.output > 1:
            # multi-classification takes the index of the largest output of softmax
            if test: axis = 1
            else: axis = 2
            y = self.dataset.classes[np.argmax(y, axis=axis)]

        elif self.dataset.classification:
            # binary classification rounds to index either 0 or 1 of classes
            y = self.dataset.classes[y.round().astype(int)]
        # regression is just the output
        performances = self.performance(y, solutions, test, eval=eval)
        return performances

    def F1(self, pred, actual):
        np.seterr(divide='ignore', invalid='ignore')
        actual = actual.astype(int)
        classes = self.dataset.classes
        timesactual = np.bincount(actual, minlength=np.max(classes) + 1)


        if len(pred.shape) > 1:
            correct = np.where(pred == actual[:, None], pred, 0).T
            timesguessed = np.apply_along_axis(np.bincount, 0, pred, minlength=np.max(classes) + 1).T
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

    @profile
    def performance(self, prediction, actual, test=False, video = False, eval=False):
        np.seterr(invalid="ignore")
        if test and eval:
            print("Predictions: ", prediction)
            print("Actuals: ", actual)
        if self.dataset.classification:
            performance = self.F1(prediction, actual)
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
    def pickWeights(self, fArray=None):
        if self.dataset.classification:
            index = np.where(np.max(fArray))
        else:
            index = np.where(np.min(fArray))

        self.tWeight = [i[index] for i in self.weights]
        self.tBWeight = [i[:, index][0] for i in self.bweights]

    # performs genetic algorithm

    def geneticAlgorithm(self, prob=.005, SD=.001, folds=10, video=False, eval=False):

        def select(fitnesses, x=5):
            # should select x number of pairs weighting selection odds by fitness
            pSelection = fitnesses ** 2 / np.sum(fitnesses ** 2)
            if video:
                print("Population Selection Probabilities:", pSelection)
            if not self.dataset.classification:
                pSelection = (1 - pSelection) / np.sum(1 - pSelection)
            pairs = [np.random.choice(np.where(fitnesses)[0], p=pSelection, replace=False, size=2) for i in range(x)]
            if video:
                print("10 Selected Parent Pairs:", pairs)
            return pairs

        def crossover(pairs):
            if video:
                print("\n\nCrossover")
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

            # parents[n][i][0] is the nth pair, the ith layer, and the weights
            if video:
                print("Sample of Parent Weights:\n")
                print("Parent 1:",parents[0][0][0][0][0,:])
                print("Parent 2:",parents[0][0][0][1][0,:])
                print("Crossover Binaries:",xover[0][0][0][0,:])
                print("Child 1:",newW[0][0][0,:])
                print("Child 2:",newW[0][10][0,:],"\n")

            return newW, newBW

        # randomly alter some genes in a given solution's chromosome, with fixed probability
        def Mutate(children, mutationProb=prob, mutationSD=SD):
            mutationBinaries = [np.random.choice([0, 1], p=[1 - mutationProb, mutationProb], size=i.shape) for i in
                                children]
            mutationTerms = [np.random.normal(0, mutationSD, size=i.shape) for i in children]
            mutated = [i + mutationBinaries[idx] * mutationTerms[idx] for idx, i in enumerate(children)]

            if video:
                print("Mutation\n")
                print("Sample of child weights pre-mutation:",children[0][0][0,:])
                print("Mutation Binaries:", mutationBinaries[0][0][0,:])
                print("Mutation Terms:", mutationTerms[0][0][0,:])
                print("Sample of child weights post-mutation:",mutated[0][0][0,:])
            return mutated

        @profile
        def run():
            fitness = self.evaluate()
            if video:
                print("Candidate Fitnesses: ", fitness)
            size = len(fitness)
            pairs = select(fitness, 10)
            newW, newBW = crossover(pairs)
            self.weights = Mutate(newW)
            if not video:
                self.bweights = Mutate(newBW)
            if self.dataset.classification:
                best = np.max(fitness)
            else:
                best = np.min(fitness)
            return best

        def train(x=500):
            performanceTrain = []
            performanceTrain.append(np.max(self.evaluate()))
            for i in range(x):
                performanceTrain.append(run())
                # print(performance[-1], gen)
            self.pickWeights(performanceTrain[-1])

        performance = []
        for i in range(folds):
            if video:
                train(x=1)
            else: train()
            perf = self.evaluate(test=True, eval=eval)
            #print("Fold %s: %s"%(i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

    def diffEvo(self, xoP=.22, sF=.04, folds=10, video = False, eval=False):

        # xoP = crossover probabiliy
        # sF = scale factor
        def mutate(fitness):
            choices = np.where(fitness)[0]
            indexes = np.array([np.random.choice(np.where(choices != i)[0], size=3, replace=False) for i in choices])
            trialWeights = [i[indexes[:, 0]] + sF * (i[indexes[:, 1]] - i[indexes[:, 2]]) for i in self.weights]
            trialBWeights = [i[:, indexes[:, 0]] + sF * (i[:, indexes[:, 1]] - i[:, indexes[:, 2]]) for i in
                             self.bweights]

            trialVectors = trialWeights, trialBWeights

            if video:
                print("Mutation (Trial Vector Generation)\n")
                print("Selected candidates from population (random):",indexes[0])
                print("Vector",indexes[0][0],":",self.weights[0][indexes[0][0]][0,:])
                print("Vector", indexes[0][1], ":", self.weights[0][indexes[0][1]][0, :])
                print("Vector", indexes[0][2], ":", self.weights[0][indexes[0][2]][0, :])
                print("Trial Vector: ",trialWeights[0][0][0,:])

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

            if video:
                print("\nCrossover (Between original and trial)\n")
                print("Original Parent:",self.weights[0][0][0,:])
                print("Trial Vector:",tV[0][0][0][0,:])
                print("Crossover Binaries:",xover[0][0][0][0,:])
                print("Child:",cW1[0][0][0,:])

            children = cW1, cBW1
            return children

        def pick(children, pFit):
            cFit = self.evaluate(children)
            keep = np.where(cFit >= pFit)
            replace = np.where(cFit < pFit)
            perf = np.min([cFit, pFit])
            if self.dataset.classification:
                keep = np.where(cFit <= pFit)
                replace = np.where(cFit > pFit)
                perf = np.max([cFit, pFit])
                if video:
                    print("\nSelection (between parent and child)\n")
                    print("Parent Fitnesses:", pFit)
                    print("Child Fitnesses:", cFit)
                    print("Parents that should be replaced by children:", np.array(replace))
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
            if self.dataset.classification:
                best = np.max(next)
            else:
                best = np.min(next)
            return best

        def train(x=500):
            performanceTrain = []
            performanceTrain.append(np.max(self.evaluate()))
            for i in range(x):
                performanceTrain.append(run())
            self.pickWeights(performanceTrain[-1])


        performance = []
        for i in range(folds):
            if video:
                train(x = 1)
            else: train()
            perf = self.evaluate(test=True, eval=eval)
            #print("Fold %s: %s"%(i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()
        performance = np.array(performance)
        return performance, performance.mean()

    def PSO(self, folds=10, video=False, eval=False):
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

            if video:
                print("\nUpdating Personal Best\n")
                print("Current Example Particle Fitness:",print(fitness[0]))
                print("Sample of Current Weights:",self.weights[0][0][0,:])
                print("Personal Best Fitness:",bestF[0])
                print("Sample of PB Weights:",bestW[0][0][0,:])

                print("New Personal Best Fitness",fitness[0])
                print("Sample of New PB Weights:",self.weights[0][0][0,:])


            return
            # ----------------
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

            # finds where, if any the current fitness is greater than the global best
            if self.dataset.classification:
                update = np.where(np.max(fitness) > gBF)
            else:
                update = np.where(fitness < gBF)

            if video:
                print("\nUpdating Global Best\n")
                print("Current Global Best Fitness:",gBF)
                print("Sample of Current gBest Position:",self.weights[0][gB][0,:])
                print("This Generation's Fitnesses:",fitness)

            # if none returns prev global best
            if len(update[0]) == 0:
                if video:
                    print("No updates to the Global Best.")
                return gB, gBF

            # if the fitness did improve sets new global best
            if self.dataset.classification:
                newGB = np.max(fitness)
            else:
                newGB = np.min(fitness)


            # sets global best index
            update = np.where(fitness == newGB)[0][0]

            if video:
                print("New Global Best Fitness:",update)
                print("Sample of New gBest position:",self.weights[0][newGB][0,:])

            return update, newGB

        # update the velocities of each particle
        def updateVelocities():

            # constriction factor, k = 1 so random walk

            phi = c*2
            k = 2 / (abs(2 - phi - np.sqrt(phi**2-4*phi)))

            r1, r2 = np.random.uniform(0, 1, size=2)

            NewV = [k*(v[idx] + c * r1 *(bestW[idx] - i) + c * r2 *(bestW[idx][gB] - i)) for idx, i in enumerate(self.weights)]
            NewBV = [k*(bv[idx] + c * r1 *(bestBW[idx] - i) + c * r2 *(bestBW[idx][:,gB] -i)) for idx, i in enumerate(self.bweights)]

            if video:
                print("\nVelocity Calculation\n")
                print("c:",c)
                print("r1, r2:",r1,"\t",r2)
                print("k:",k)
                print("Sample of Previous Velocity Values:",v[0][0][0,:])
                print("Sample of New Velocity Values:",NewV[0][0][0,:])

            return NewV, NewBV

            # --------------------
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
        def updatePositions(v, bv):

            if video:
                print("\nUpdating Positions\n")
                print("Sample of Previous Position:",self.weights[0][0][0,:])
                print("Sample of Particle Velocity:",v[0][0][0,:])
            self.weights = [i+v[idx] for idx, i in enumerate(self.weights)]
            self.bweights = [i+bv[idx] for idx, i in enumerate(self.bweights)]
            if video:
                print("New Position:",self.weights[0][0][0,:])
            return

            # --------------------

            # for each particle
            for i in range(len(pBests)):

                # add the velocity of each particle onto the current position
                [self.weights[0][i], self.weights[1][i], self.weights[2][i]] = [j[i] + k for j, k in zip(self.weights, velocities[i])]

            return self.weights

        def run():
            fitness = self.evaluate()
            updatePBests(fitness)
            globalBest, globalBestFitness = updateGBest(fitness)
            newV, newBv = updateVelocities()
            updatePositions(newV, newBv)
            return globalBest, globalBestFitness, newV, newBv

        def train(x = 500):
            nonlocal gB, gBF, v, bv
            gB, gBF = updateGBest(bestF)
            performanceTrain = []
            performanceTrain.append(gBF)
            for i in range(x):
                gB, gBF, v, bv = run()
            self.pickWeights(bestF)


        performance = []
        for i in range(folds):
            if video:
                train(x=1)
            else: train()
            perf = self.evaluate(test=True, eval=eval)
            #print("Fold %s: %s" % (i + 1, perf))
            performance.append(perf)
            self.reset()
            self.dataset.nextFold()


        performance = np.array(performance)
        return performance, performance.mean()



#Sample Outputs from one test fold for one regression, one classification, two hidden layers, all learning methods

glass = Dataset("glass")
#Glass dataset
#GA
print("Classification: Glass Dataset \n")
print("Genetic Algorithm")
output = glass.networks[2].geneticAlgorithm(.2, .02, folds=1, eval=True)
print("Mean F1, 1 Fold:", output[1])
#DE
print("\nDifferential Evolution")
output = glass.networks[2].diffEvo(.005, .005, folds=1, eval=True)
print("Mean F1, 1 Fold:", output[1])

#PSO
print("\nPSO")
output = glass.networks[2].PSO(folds=1, eval=True)
print("Mean F1, 1 Fold:", output[1])

print("\n\nRegression: Machine Dataset \n")
#Machine dataset
machine = Dataset("machine")
#GA
print("Genetic Algorithm")
output = machine.networks[2].geneticAlgorithm(.01, .001, folds=1, eval=True)
print("Mean MSE, 1 Fold:", output[1])
#DE
print("\nDifferential Evolution")
output = machine.networks[2].diffEvo(.4, .2, folds=1, eval=True)
print("Mean MSE, 1 Fold:", output[1])
#PSO
print("\nPSO")
output = machine.networks[2].PSO(folds=1, eval=True)
print("Mean MSE, 1 Fold:", output[1])

#GA Operations
print("\n\nGenetic Algorithm Operations\n\n")
#Selection
print("Selection\n")
glass.networks[2].geneticAlgorithm(.01, .001, folds=1, video=True)


#DE Operations
print("\n\nDifferential Evolution Operations\n\n")
print("Selection\n")
glass.networks[2].diffEvo(.005, .005, folds=1, video=True)

#PSO Operations
print("\n\nParticle Swarm Optimization Operations\n\n")
glass.networks[2].PSO(folds=1, video=True)

#Average performance over 10 folds for one of the datasets for each of the networks