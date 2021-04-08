#import statements
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import inspect
import sys
# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn

#genarrayav=[]
genarraymin=[]
genarrayav=[]
#mutd_values_CGA =[] #store the mutated values
#mutd_values_GA =[] #store the mutated values

#Histogram plotting function
def plotHistogram(map_array, name):


    n, bins, patches = plt.hist(x=map_array, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)

    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

# Logistic Class
class LM:
    def __init__(self, x_0, r, shift, scale):
        self.x = x_0
        self.r = r
        self.shift= shift
        self.scale = scale

    def next_val(self):
        self.x = self.r * self.x *(1.0-self.x)
        return self.x

    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale

# Gaussian class - for GA
class Gauss:
    def __init__(self, shift, scale):
        self.shift= shift
        self.scale = scale

    def next_val(self):
        return np.random.normal()

    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale

def logisticMap(r, x_n):
    return r * x_n *(1.0-x_n)

def Rosenbrock(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for elem in range(1,d-1):
        sigma+=(100*((array[elem+1]-(array[elem])**2)**2)+(array[elem]-1)**2)
    return sigma

def Griewank(array):

    #summation, sigma
    sigma=0
    #product, pi
    pi=1
    # length of the array
    d = len(array)

    for val in range(1,d):
        sigma+=math.pow(array[val],2)
        pi*=math.cos(float(array[val])/math.sqrt(val))

    return (float(sigma)/4000)-float(pi)+1


def Rastrigin(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for el in range(1,d):
        sigma+=((array[el]**2)-(10*math.cos(2*math.pi*array[el])))

    return (10*d)+sigma

def generatePopulation(obj, pop_size, indiv_size):
    population =[]
    for i in range(pop_size):
        values=[]
        for i in range(indiv_size):
            values.append(obj.shift_scale_next())
        population.append([values,math.inf])
    return population



#.           [0 ]                       [1]
#       0                    1        0     1
#[[[array of 100 LM values],inf], [[array],inf]...]
def evaluate_fitness(population,benchMark):

    fitarray=[]

    for index in range(len(population)):


        if benchMark == Rastrigin:
            bm_value = Rastrigin(population[index][0])
        elif benchMark == Rosenbrock:
            bm_value = Rosenbrock(population[index][0])
        else:
            bm_value = Griewank(population[index][0])

        fitarray.append(bm_value)

        # changes the fitness value
        population[index][1]=bm_value
    average = sum(fitarray)/len(fitarray)#average fitness of individuals at some generation
    genarrayav.append(average)
    genarraymin.append(min(fitarray))
    #genarraymax.append(max(fitarray))
    #print(len(genarrayav))


def find_fittest(populat):
    min = populat[0][1]
    winnerIndividual=populat[0]

    for item in range(len(populat)):
        if populat[item][1]< min:
            min=populat[item][1]
            winnerIndividual=populat[item]

    return winnerIndividual

def get_parent(pop):
    #print("In pick Parent")
    lengthOfPop = len(pop)
    #print("The length is {}".format(lengthOfPop))
    randomItem1 = random.randint(0,lengthOfPop-1)
    parentA=pop[randomItem1][0]
    #print("Parent A is {}".format(parentA))
    Afit = pop[randomItem1][1]
    #print("Parent A's fitness is {}".format(Afit))

    randomItem2 = random.randint(0,lengthOfPop-1)
    parentB=pop[randomItem2][0]
    #print("Parent B is {}".format(parentB))
    Bfit = pop[randomItem2][1]
    #print("Parent B's fitness is {}".format(Afit))

    if Afit < Bfit:
        #print("{} is < than {}, so return {}".format(Afit,Bfit,Afit))
        return parentA
    else:
        #print("{} is < than {}, so return {}".format(Bfit,Afit,Bfit))
        return parentB

def crossover(p1,p2,probability):
    #print("In Crossover Function... \n parent1 = {}\n parent2 ={}".format(p1,p2))

    parentLen = len(p1)
    num = random.uniform(0.0,1.0)
    if num < probability:
        crossPoint= random.randint(0,parentLen)
        #print("crosspoint={}".format(crossPoint))
        newKid1= p1[:crossPoint]
        newKid2= p2[crossPoint:]
        newKid = newKid1+newKid2
        #print("The new Kid is {}".format(newKid))
        return newKid
    else:
        #print("No crossover, so returning p1 = {}".format(p1))
        return copy.deepcopy(p1)


def mutate(individual,probability,algorithm_object):


    #for each individual's genes, get a random number between 0 and 1
    for gene in range(len(individual)):
        num = random.uniform(0.0,1.0)
        if num < probability:
            n_val=algorithm_object.shift_scale_next()
            individual[gene]+=n_val

    return individual


# THE GA DRIVER

#generate the initial population  [[[array of 100 LM values],inf], [,]...] //total = 200
'''
This is a simple function/code that calculates the average and std. dev
for n trails of running the GA/CGA
'''




#Use variables & make code more flexible


#map = lm
probabilitym = 0.05
probabilityc = 0.8
gen_size = 10000
default_fitness= math.inf
num_trails=10
population_size=50
individual_size=20
lm = LM(0.02, 4.0,-0.5,2)
rdm = Gauss(0,0.5)

def EA(map,gen_size,probabilitym,default_fitness,pop,probabilityc,bm):


    #evaluate the fitness
    evaluate_fitness(pop,bm)

    #get the fittest individual
    fittest = find_fittest(pop)

    #this is the first (zeroth) generation
    gen = 0

    # so far within constraints
    while fittest[1]>0 and gen <gen_size:
        # the best is the new pop -> may/may not do this | elitism
        new_population = [fittest]

        #[g,g,g,g,g,g]
        #add to the new population:
        for i in range(len(pop)-1):
            #pick 2 parents
            p1 = get_parent(pop)
            p2 = get_parent(pop)
            #crossover
            potential_child = crossover(p1,p2,probabilityc)
            #potential_child = copy.deepcopy(p1)
            #mutation
            #print("\t\tpotential child is: {}".format(potential_child))
            child = mutate(potential_child,probabilitym,map)
            new_population.append([child, default_fitness])

        #make the new-population the next pop to work with
        pop = new_population

        #generation increases by 1
        gen+=1

        #evaluate the fitness of this population
        evaluate_fitness(pop,bm)

        #make the best
        fittest = find_fittest(pop)
        #print("Generation {}: Fittest: {}".format(gen,fittest))
    print("Ran Successfully!")


def getMinAndAvgFitness(xZero,lambda_val,l_min_fitness_array,l_avg_fitness_array,benchmark):
    genarraymin.clear()
    genarrayav.clear()

    lm = LM(xZero, lambda_val,-0.5,2)
    rdm = Gauss(0,0.5)
    pop = generatePopulation(rdm, population_size, individual_size)
    EA(lm,gen_size,probabilitym,default_fitness,pop,probabilityc,benchmark)

    mn = genarraymin.copy()
    av = genarrayav.copy()

    #print("mn array is {}".format(mn))
    l_min_fitness_array.append(mn)
    l_avg_fitness_array.append(av)


def plotMapParameters(l1,l2,l3,l4,l5,benchmark):
    #list containing proposed x0s
    xZeros=[0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.497]

    '''
    Run with each of these values and run it with those results for 10-20 trails.
    Similar to the GA2.py file.
    '''

    '''
    Run with average and pick a random x0 each time to plot it. (Like GA2.py for every trail.)

    Then run the CGA VS GA with it.
    EXPECTATION: Should be the same.
    '''




    #store the values to plot
    l1_minFitness =[]
    l2_minFitness =[]
    l3_minFitness =[]
    l4_minFitness =[]
    l5_minFitness =[]

    l1_avgFitness =[]
    l2_avgFitness =[]
    l3_avgFitness =[]
    l4_avgFitness =[]
    l5_avgFitness =[]

    l1_minCombined =[]
    l2_minCombined =[]
    l3_minCombined =[]
    l4_minCombined =[]
    l5_minCombined =[]

    l1_avgCombined =[]
    l2_avgCombined =[]
    l3_avgCombined =[]
    l4_avgCombined =[]
    l5_avgCombined =[]

    #print("Before run, the l fitness array is {}".format(l1_minFitness))

    #run for 20 trails ...
    for idx in range(num_trails):
        getMinAndAvgFitness(xZeros[idx],l1,l1_minFitness,l1_avgFitness,benchmark)
        #print("Min fitness array with x0: {} and lambda {} of {} size is {}\n\n".format(xZeros[0],l1,l1_minFitness,len(l1_minFitness)))
        getMinAndAvgFitness(xZeros[idx],l2,l2_minFitness,l2_avgFitness,benchmark)
        getMinAndAvgFitness(xZeros[idx],l3,l3_minFitness,l3_avgFitness,benchmark)
        getMinAndAvgFitness(xZeros[idx],l4,l4_minFitness,l4_avgFitness,benchmark)
        getMinAndAvgFitness(xZeros[idx],l5,l5_minFitness,l5_avgFitness,benchmark)




    #convert to np ARRAYS
    l1_minFitnessAvg = np.mean(np.array(copy.deepcopy(l1_minFitness)),axis=0)
    l2_minFitnessAvg = np.mean(np.array(copy.deepcopy(l2_minFitness)),axis=0)
    l3_minFitnessAvg = np.mean(np.array(copy.deepcopy(l3_minFitness)),axis=0)
    l4_minFitnessAvg = np.mean(np.array(copy.deepcopy(l4_minFitness)),axis=0)
    l5_minFitnessAvg = np.mean(np.array(copy.deepcopy(l5_minFitness)),axis=0)

    l1_avgFitnessAvg = np.mean(np.array(copy.deepcopy(l1_avgFitness)),axis=0)
    l2_avgFitnessAvg = np.mean(np.array(copy.deepcopy(l2_avgFitness)),axis=0)
    l3_avgFitnessAvg = np.mean(np.array(copy.deepcopy(l3_avgFitness)),axis=0)
    l4_avgFitnessAvg = np.mean(np.array(copy.deepcopy(l4_avgFitness)),axis=0)
    l5_avgFitnessAvg = np.mean(np.array(copy.deepcopy(l5_avgFitness)),axis=0)


    l1_minCombined.append(l1_minFitnessAvg)
    l2_minCombined.append(l1_minFitnessAvg)
    l3_minCombined.append(l1_minFitnessAvg)
    l4_minCombined.append(l1_minFitnessAvg)
    l5_minCombined.append(l1_minFitnessAvg)

    l1_avgCombined.append(l1_avgFitnessAvg)
    l2_avgCombined.append(l2_avgFitnessAvg)
    l3_avgCombined.append(l3_avgFitnessAvg)
    l4_avgCombined.append(l4_avgFitnessAvg)
    l5_avgCombined.append(l5_avgFitnessAvg)



    l1_minCombined = np.mean(np.array(copy.deepcopy(l1_minCombined)),axis=0)
    l2_minCombined = np.mean(np.array(copy.deepcopy(l2_minCombined)),axis=0)
    l3_minCombined = np.mean(np.array(copy.deepcopy(l3_minCombined)),axis=0)
    l4_minCombined = np.mean(np.array(copy.deepcopy(l4_minCombined)),axis=0)
    l5_minCombined = np.mean(np.array(copy.deepcopy(l5_minCombined)),axis=0)

    l1_avgCombined = np.mean(np.array(copy.deepcopy(l1_avgCombined)),axis=0)
    l2_avgCombined = np.mean(np.array(copy.deepcopy(l2_avgCombined)),axis=0)
    l3_avgCombined = np.mean(np.array(copy.deepcopy(l3_avgCombined)),axis=0)
    l4_avgCombined = np.mean(np.array(copy.deepcopy(l4_avgCombined)),axis=0)
    l5_avgCombined = np.mean(np.array(copy.deepcopy(l5_avgCombined)),axis=0)



    #Let's plot
    x = [i for i in range(gen_size+1)]

    #plt.plot(x,l1_minFitnessAvg,color = "blue")


    plt.plot(x,l1_minCombined,color="red",label=l1)
    plt.plot(x,l2_minCombined,color="orange",label=l2)
    plt.plot(x,l3_minCombined,color="green",label=l3)
    plt.plot(x,l4_minCombined,color="blue",label=l4)
    plt.plot(x,l5_minCombined,color="purple",label=l5)


    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('Logistic Map on {} Fitness Function, Min Fitness Combined'.format(benchmark.__name__))

    # show a legend on the plot
    plt.legend()
    plt.show()


    # PLOTS FOR AVERAGE FITNESS
    x = [i for i in range(gen_size+1)]

    #plt.plot(x,l1_minFitnessAvg,color = "blue")


    plt.plot(x,l1_avgCombined,color="red",label=l1)
    plt.plot(x,l2_avgCombined,color="orange",label=l2)
    plt.plot(x,l3_avgCombined,color="green",label=l3)
    plt.plot(x,l4_avgCombined,color="blue",label=l4)
    plt.plot(x,l5_avgCombined,color="purple",label=l5)

    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('Logistic Map on {} Fitness Function , Average Fitness Combined'.format(benchmark.__name__))

    # show a legend on the plot
    plt.legend()
    plt.show()

    '''

    #CGA vs GA comparison
    for i in range(num_trails):
        genarrayav.clear()
        genarraymin.clear()

        pop = generatePopulation(rdm, population_size, individual_size)

        pop_GA = copy.deepcopy(pop)
        pop_CGA = copy.deepcopy(pop)

        EA(lm,gen_size,probabilitym,default_fitness,pop_CGA,probabilityc,benchmark)
        av = genarrayav.copy()
        mn = genarraymin.copy()

        genarrayav.clear()
        genarraymin.clear()

        EA(rdm,gen_size,probabilitym,default_fitness,pop_GA,probabilityc,benchmark)

        avGA = genarrayav.copy()
        mnGA = genarraymin.copy()

        avgs2D_CGA.append(av)
        mins2D_CGA.append(mn)

        avgs2D_GA.append(avGA)
        mins2D_GA.append(mnGA)

        avgs2D_CGA = np.array(avgs2D_CGA)
        mins2D_CGA = np.array(mins2D_CGA)
        avgs2D_GA = np.array(avgs2D_GA)
        mins2D_GA = np.array(mins2D_GA)

        avg_avgs_CGA = np.mean(avgs2D_CGA,axis=0)
        avg_mins_CGA = np.mean(mins2D_CGA,axis=0)
        avg_avgs_GA = np.mean(avgs2D_GA,axis=0)
        avg_mins_GA = np.mean(mins2D_GA,axis=0)

        x = [i for i in range(gen_size+1)]
        #REGULAR PLOTS AVG-AVG
        plt.plot(x, avg_avgs_CGA,  label = "average-fitness of averages (CGA)")
        plt.plot(x, avg_avgs_GA,  label = "average-fitness of averages (GA)")
        # naming the x axis
        plt.xlabel('generation')
        # naming the y axis
        plt.ylabel('fitness')
        # giving a title to my graph
        plt.title('Average of Average Fitness per Generation')

        # show a legend on the plot
        plt.legend()
        plt.show()

        #REGULAR PLOTS AVG-MINS
        plt.plot(x, avg_mins_CGA,   label = "average-fitness of mins (CGA)")
        plt.plot(x, avg_mins_GA,  label = "average-fitness of mins (GA)")

        # naming the x axis
        plt.xlabel('generation')
        # naming the y axis
        plt.ylabel('fitness')
        # giving a title to my graph
        plt.title('Average of Min Fitness per Generation')

        # show a legend on the plot
        plt.legend()
        plt.show()
        '''









plotMapParameters(3.6,3.7,3.8,3.9,4.0,Rastrigin)
#plotMapParameters(3.8,3.9,4.0,Rosenbrock)
#plotMapParameters(3.8,3.9,4.0,Griewank)
