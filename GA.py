import math
import random
import matplotlib.pyplot as plt
import numpy as np

# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn

genarrayav=[]
genarraymax=[]
genarraymin=[]



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

def rosenbrock(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for elem in range(1,d-1):
        sigma+=(100*((array[elem+1]-(array[elem])**2)**2)+(array[elem]-1)**2)
    return sigma

def generatePopulation(obj):
    population_size = 5
    individual_size = 5
    population =[]
    for i in range(population_size):
        values=[]
        for i in range(individual_size):
            values.append(obj.shift_scale_next())
        population.append([values,math.inf])
    return population



#.           [0 ]                       [1]
#       0                    1        0     1
#[[[array of 100 LM values],inf], [[array],inf]...]
def evaluate_fitness(population):

    fitarray=[]

    for index in range(len(population)):

        bm_value = rosenbrock(population[index][0])
        fitarray.append(bm_value)#should get 200 bms
        population[index][1]=bm_value
    average = sum(fitarray)/len(fitarray)
    genarrayav.append(average)
    genarraymin.append(min(fitarray))
    genarraymax.append(max(fitarray))


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

def crossover(p1,p2):
    parentLen = len(p1)
    crossPoint= random.randint(0,parentLen)
    newKid1= p1[:crossPoint]
    newKid2= p2[crossPoint:]
    newKid = newKid1+newKid2
    return newKid


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

def dev_and_avg(num_trails):
    #stats_array to store the array of arrays used to calculate the average and standard deviation
    stats_array_avgavg=[]
    stats_array_minavg=[]
    deviation_min=[]
    deviation_avg=[]

    # run the EA according to the number of trails
    # and store each min and average array in stats_array
    for _ in range(num_trails):
        EA(lm,rdm,map,gen_size,probability,default_fitness)
        stats_array_minavg.append(np.average(genarraymin))
        stats_array_avgavg.append(np.average(genarrayav))
        deviation_min.append(np.std(genarraymin))
        deviation_avg.append(np.std(genarrayav))
    #Note: there is already an array of avs and mins at each run

    #plot against trails
    x = [i for i in range(num_trails)]
    y = stats_array_avgavg
    # plotting the line 1 points
    plt.plot(x, y, label = "average-fitness of averages")

    xx = [i for i in range(num_trails)]
    yy = stats_array_minavg
    # plotting the line 1 points
    plt.plot(xx, yy, label = "average-fitness of mins")

    xo = [i for i in range(num_trails)]
    yo = deviation_avg
    plt.plot(xo, yo, label = "standard deviation of avgs")

    xo1 = [i for i in range(num_trails)]
    yo1 = deviation_min
    plt.plot(xo1, yo1, label = "standard deviation of mins")


    plt.xlabel('trails')
    plt.ylabel('fitness')
    plt.title('Average and Standard Deviation per Generation trails')
    plt.legend()
    plt.show()

#Use variables & make code more flexible
lm = LM(0.01, 3.8,-0.5,2)
rdm = Gauss(0,1)
map = lm
probability = 0.01
gen_size = 500
default_fitness= math.inf
num_trails=100

def EA(lm,rdm,map,gen_size,probability,default_fitness):


    #generate the initial population
    pop = generatePopulation(map)


    #evaluate the fitness
    evaluate_fitness(pop)

    #get the fittest individual
    fittest = find_fittest(pop)

    #this is the first (zeroth) generation
    gen = 0

    # so far within constraints
    while fittest[1]>0 and gen <gen_size:
        # the best is the new pop -> may/may not do this | elitism
        new_population = [fittest]

        #add to the new population:
        for i in range(len(pop)-1):
            #pick 2 parents
            p1 = get_parent(pop)
            p2 = get_parent(pop)
            #crossover
            potential_child = crossover(p1,p2)
            #mutation
            '''
            How can you call lm to mutate?
            Note: initial x_0 is gone
            Usual probablility is [0.005,0.02]
            too high of a mutation makes it hard to converge(exploit) and
            more exploration
            '''
            child = mutate(potential_child,probability,map)
            new_population.append([child, default_fitness])

        #make the new-population the next pop to work with
        pop = new_population

        #generation increases by 1
        gen+=1

        #evaluate the fitness of this population
        evaluate_fitness(pop)

        #make the best
        fittest = find_fittest(pop)
        print("Generation {}: Fittest: {}".format(gen,fittest))
    print("Ran Successfully!")


    '''
    #Plot Stuff
    # average fitness points
    x1 = [x for x in range(gen+1)]
    y1 = genarrayav
    # plotting the line 1 points
    plt.plot(x1, y1, label = "average-fitness")



    # min fitness points
    x2 = [x for x in range(gen+1)]
    y2 = genarraymin
    # plotting the line 2 points
    plt.plot(x2, y2, label = "min-fitness")

    # naming the x axis
    plt.xlabel('generation')
    # naming the y axis
    plt.ylabel('fitness')
    # giving a title to my graph
    plt.title('Average and Min Fitness per Generation')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    #test git --> 10/31/2020 11:15pm
    #second test --> 11/02/2020 00:03am
    plt.show()
    '''

# RUN THE GA
#standard deviation and Average
dev_and_avg(num_trails)
#EA(lm,rdm,map,gen_size,probability,default_fitness)
