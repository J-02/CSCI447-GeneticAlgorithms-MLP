import numpy as np
from MLP import MLP

#GA Experimentation Pseudocode; Trains a population of MLPs and uses a GA to select the best one

class GAModel:
    #initializes with a population of solutions, a probability of crossover, a probability of mutation, and a mutation SD
    def __init__(self, solutionPopulation = [], probOfCrossover = .1, probOfMutation = .05, mutationSD = .01):
        self.solutionPopulation = solutionPopulation
        self.populationSize = len(solutionPopulation)
        self.populationFitnesses = []
        self.probOfCrossover = probOfCrossover
        self.probOfMutation = probOfMutation
        self.mutationSD = mutationSD

    #training function, iterates through generations until the best model in a given generation doesn't improve sufficiently
    #from previous generation
    def Train(self):
        converged = False
        maxSolutionFitness = 0

        while not converged:
            for model in self.solutionPopulation:
                #will need to modify training of MLP to not back-propagate, and return preds/actuals
                #calculate fitnesses of all models in population
                self.populationFitnesses.append(self.performance(model.Train()))

            #performs n/5 reproductions, will probably need to figure out how to decide this later
            newSolutionPopulation = []
            newPopulationFitnesses = []
            for i in range(int(self.populationSize/5)):


                #easier to return the indices in the list of the parents selected
                parent1, parent2 = self.selectParents()

                #crossover the 2 selected parents
                child1, child2 = self.crossParents(self.solutionPopulation[parent1], self.solutionPopulation[parent2])

                #randomly mutate some genes in each child
                child1 = self.Mutate(child1)
                child2 = self.Mutate(child2)


                #using generational replacement, can adjust to steady state later if we want

                #remove parents from original population so not selected multiple times (should they be able to be
                # selected multiple times?)
                self.solutionPopulation.remove(parent1)
                self.solutionPopulation.remove(parent2)

                #add children to the "next generation"
                newSolutionPopulation.append(child1)
                newSolutionPopulation.append(child2)

                #add children performances to the "next generation"
                newPopulationFitnesses.append(self.performance(child1.Train()))
                newPopulationFitnesses.append(self.performance(child2.Train()))

            #set the current generation population to the newly generated population of children
            self.solutionPopulation = newSolutionPopulation
            self.populationFitnesses = newPopulationFitnesses

            #continue until the best solution in the children population doesn't improve that much, and return
            #that solution
            maxSolutionFitness = max(self.populationFitnesses)
            bestSolution = self.solutionPopulation[np.argmax(self.populationFitnesses)]

            #need to add convergence condition here
        return bestSolution, maxSolutionFitness


    def Test(self):
        pass



    #select 2 parents to reproduce and create 2 children
    def selectParents(self):
        #select 2 parents probabilistically based on relative fitnesses
        #probably involves summing all fitnesses and calculating probabilities of each parent being selected
        parent1, parent2 = 0
        return parent1, parent2

    #cross parents over, uniform crossover (with 10% probability) implemented but could switch to one point or two point
    def crossParents(self, parent1, parent2):

        #need to flatten weight matrices into 1d arrays to get into "chromosome" form
        #not sure syntactically the best way to do this, can think of some shitty ways though
        parentChromosome1 = np.flatten(parent1.weights)
        parentChromosome2 = np.flatten(parent2.weights)

        childChromosome1 = parentChromosome1
        childChromosome2 = parentChromosome2

        #iterate through all genes in the chromosomes
        for i in len(parentChromosome1):

            #swap the genes from the two parents in the children with some fixed probability
            chanceOfCrossover = np.random.uniform(0,1)
            if chanceOfCrossover < self.probOfCrossover:
                childChromosome1[i] = parentChromosome2[i]
                childChromosome2[i] = parentChromosome1[i]

        #return children as new solutions with chromosomes as weight matrix, all else held the same
        # child1, child2 = MLP(add appropriate MLP parameters here, might need a setter for weight matrices)
        child1, child2 = 0
        return child1, child2


    #randomly alter some genes in a given solution's chromosome, with fixed probability
    def Mutate(self, solution):

        #again, probably an optimal way to get to this 1d array, or might not even be necessary
        chromosome = np.flatten(solution.weights)

        #iterate through all genes, mutate some % with a term from a normal distribution with fixed SD (need to tune)
        for i in len(chromosome):
            chanceOfMutation = np.random.uniform(0,1)
            if chanceOfMutation < self.probOfMutation:
                chromosome[i] = chromosome[i] + np.random.normal(0, self.mutationSD)

        mutatedSolution = MLP(chromosome)


    def performance(self, solution):
        fitness = 0
        return fitness

