import math
import random
import matplotlib.pyplot as plt
import numpy as np

# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn

genarrayav=[]
genarraymin=[]
mutd_values =[] #store the mutated values

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

def rosenbrock(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for elem in range(1,d-1):
        sigma+=(100*((array[elem+1]-(array[elem])**2)**2)+(array[elem]-1)**2)
    return sigma

def griewank(array):

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


def rastrigin(array):
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
def evaluate_fitness(population):

    fitarray=[]

    for index in range(len(population)):

        bm_value = rosenbrock(population[index][0])
        fitarray.append(bm_value)
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
            print("\nAdding {} to the mutated array....".format(n_val))
            mutd_values.append(n_val)
            print("mutd_array now contains {} values with the first value ->{} and the last value-> {}".format(len(mutd_values), mutd_values[0], mutd_values[-1]))
            individual[gene]+=n_val

    return individual


# THE GA DRIVER

#generate the initial population  [[[array of 100 LM values],inf], [,]...] //total = 200
'''
This is a simple function/code that calculates the average and std. dev
for n trails of running the GA/CGA
'''




#Use variables & make code more flexible
lm = LM(0.01, 3.8,-0.5,2)
rdm = Gauss(0,1)
#map = lm
probability = 0.01
gen_size = 10
default_fitness= math.inf
num_trails=5
population_size=15
individual_size=15

def EA(map,gen_size,probability,default_fitness):


    #generate the initial population
    pop = generatePopulation(rdm, population_size, individual_size)


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
        #print("Generation {}: Fittest: {}".format(gen,fittest))
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



def plots():
    #mean and standard deviation plots
    avgs2D_CGA=[]
    mins2D_CGA=[]

    avgs2D_GA=[]
    mins2D_GA=[]


    """
    IDEA: Statistically start from the same population
    Approach #1:
    New trail will be one run of CGA & GA, where we start at the same initial population (still Gaussian? LM?).
    - will change outside the EA method
    - gen pop
    - make copies
    - use one copy for the GA
    - use one copy for the CGA
    • guarantees the performance on the same init pop

    Approach #2- THIS IS THE APPROACH USED IN THIS  FILE:
    Gen the population using Gaussian rather than LM
    - will change within the EA method. Every trail will have a different init pop, but generated the same way.
    • possible, but unlikely that the GA gets bad pops (pop picked when EA is ran, the pop made the GA perform poorly)
     and the CGA gets good pops over some number of trails.

    HOW to know which approach is better than the other?
    Do either approach reduce the likelihood of one of the algorithms looking better by chance?
    We don't want a set of results to show one better than the other because of the initial pops picked.
    Need a fair expeiment- truth vs chance

    • Multiple trails can reduce the chance

    """
    #DEALS WITH CGA
    for i in range(num_trails):

        #reset
        print("\n\nClearing the arrays for generating the averages and mins")
        genarrayav.clear()
        genarraymin.clear()
        print("The arrays are now: ")
        print(genarrayav)
        print(genarraymin)

        print("Running the {} instance of the CGA".format(i+1))
        EA(lm,gen_size,probability,default_fitness)# Run the CGA
        #print("The avg array has {} elements (should be 500 ish)".format(len(genarrayav)))
        print("\nGenArrayAv for CGA trail {}:".format(i+1))
        print(genarrayav)
        av = genarrayav.copy()
        print("GenArrayMin for CGA trail {}:".format(i+1))
        print(genarraymin)
        mn = genarraymin.copy()

        print("Before appending to the 2Ds, the arrays contain:\navgs2D_CGA:")
        print(avgs2D_CGA)
        print("mins2D_CGA:")
        print(mins2D_CGA)

        print("Appending genarrayav and genarraymin to the avgs2D_GA and mins2D_GA respectively...")
        avgs2D_CGA.append(av) # add the average fitness across generations
        print("avgs2D_CGA now contains...")
        print(avgs2D_CGA)
        mins2D_CGA.append(mn)# add the min fitness across generations to the 2D array
        print("mins2D_CGA now contains...")
        print(mins2D_CGA)

    CGA_mutd_values = mutd_values.copy()
    print("\n\n\nTHE SHIFT SCALE VALUES ARE:\nCGA:")
    print(CGA_mutd_values)
    print("clearing the mutd array......................................")
    mutd_values.clear()

    #DEALS WITH GA
    for j in range(num_trails):

        #reset
        print("\n\nClearing the arrays for generating the averages and mins")
        genarrayav.clear()
        genarraymin.clear()
        print("The arrays are now: ")
        print(genarrayav)
        print(genarraymin)

        print("Running the {} instance of the GA".format(j+1))
        EA(rdm,gen_size,probability,default_fitness)# Run the GA
        #print("The avg array has {} elements (should be 500 ish)".format(len(genarrayav)))
        print("\nGenArrayAv for GA trail {}:".format(j+1))
        print(genarrayav)
        av = genarrayav.copy()

        print("GenArrayMin for GA trail {}:".format(j+1))
        print(genarraymin)
        mn = genarraymin.copy()



        print("Before appending to the 2Ds, the arrays contain:\navgs2D_GA:")
        print(avgs2D_GA)
        print("mins2D_GA:")
        print(mins2D_GA)


        print("Appending genarrayav and genarraymin to the avgs2D_GA and mins2D_GA respectively...")
        avgs2D_GA.append(av) # add the average fitness across generations
        print("avgs2D_GA now contains...")
        print(avgs2D_GA)
        mins2D_GA.append(mn)# add the min fitness across generations to the 2D array
        print("mins2D_GA now contains...")
        print(mins2D_GA)

    GA_mutd_values = mutd_values.copy()

    #HERE! - PRINT THE ARRAYS (WITH A SMALL SET) & COMPARE THEM
    print("\nTHE 2D ARRAYS FOR GA AND CGA: \nCGA:\nAVERAGES-")
    print(avgs2D_CGA)
    print("MINS-")
    print(mins2D_CGA)

    print("\nGA:\nAVERAGES-")
    print(avgs2D_GA)
    print("MINS-")
    print(mins2D_GA)

    print("\n\n\nTHE SHIFT SCALE VALUES ARE:\nCGA:")
    print(CGA_mutd_values)
    print("GA:")
    print(GA_mutd_values)

    #at the end, should have two 2D arrays
    #use the np mean and std dev to compute them in an array
    avgs2D_CGA = np.array(avgs2D_CGA)
    mins2D_CGA = np.array(mins2D_CGA)
    avgs2D_GA = np.array(avgs2D_GA)
    mins2D_GA = np.array(mins2D_GA)



    std_avgs_CGA = np.std(avgs2D_CGA,axis=0)
    std_mins_CGA = np.std(mins2D_CGA,axis=0)
    avg_avgs_CGA = np.mean(avgs2D_CGA,axis=0)
    avg_mins_CGA = np.mean(mins2D_CGA,axis=0)


    #GA
    std_avgs_GA = np.std(avgs2D_GA,axis=0)
    std_mins_GA = np.std(mins2D_GA,axis=0)
    avg_avgs_GA = np.mean(avgs2D_GA,axis=0)
    avg_mins_GA = np.mean(mins2D_GA,axis=0)
    #print(len(avgs2D[0]))
    #print(len(avg_mins))
    #print(std_avgs)
    #print(avg_avgs)
    #plot for averages
    x = [i for i in range(gen_size+1)]
    #x2 = [i for i in range(gen_size)]
    """
    plt.errorbar(x, avg_avgs_CGA, std_avgs_CGA, label = "average-fitness of averages (CGA)")
    plt.errorbar(x, avg_mins_CGA, std_mins_CGA,  label = "average-fitness of mins (CGA)")
    plt.errorbar(x, avg_avgs_GA, std_avgs_GA,  label = "average-fitness of averages (GA)")
    plt.errorbar(x, avg_mins_GA, std_mins_GA,  label = "average-fitness of mins (GA)")
    """

    #SUBPLOTS FOR ANG AND MIN EAs
    fig,ax = plt.subplots(2,2)
    ax[0,0].errorbar(x, avg_avgs_GA)
    ax[0,0].set_title('Average of Average Fitness per Genaration (GA)')
    ax[0,1].errorbar(x, avg_mins_GA,'tab:orange')
    ax[0,1].set_title('Average of Min Fitness per Genaration (GA)')
    ax[1,0].errorbar(x, avg_avgs_CGA,'tab:green')
    ax[1,0].set_title('Average of Average Fitness per Genaration (CGA)')
    ax[1,1].errorbar(x, avg_mins_CGA,'tab:red')
    ax[1,1].set_title('Average of Min Fitness per Genaration (CGA)')

    for ax in ax.flat:
        ax.set(xlabel='generation', ylabel='fitness')
        ax.label_outer()


    plt.show()

    #SUBPLOTS FOR DISTRIBUTIONS

    plotHistogram(CGA_mutd_values,'Shift-Scale Distributions for CGA')

    plotHistogram(GA_mutd_values,'Shift-Scale Distributions for GA')

    figs,axs = plt.subplots(2,1)
    for ax in axs.flat:
        ax.set(xlabel='values', ylabel='frequency')
        ax.label_outer()

    axs[0,0]= plotHistogram(CGA_mutd_values,'Shift-Scale Distributions for CGA')
    axs[1,0]= plotHistogram(GA_mutd_values,'Shift-Scale Distributions for GA')
    plt.show()

    '''
    n, bins, patches = axs[0,0].hist(x=CGA_mutd_values, bins=20, color='#0504aa',alpha=0.7, rwidth=0.85)
    axs[0,0].set_title('Shift-Scale Distributions for CGA')
    axs[0,0].grid(axis='y', alpha=0.75)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    axs[0,0].ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    n, bins, patches = axs[1,0].hist(x=GA_mutd_values, bins=20, color='#0504aa',alpha=0.7, rwidth=0.85)
    axs[1,0].grid(axis='y', alpha=0.75)
    axs[1,0].set_title('Shift-Scale Distributions for GA')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    axs[1,0].ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    '''




    #plt.show()

    '''
    plt.xlabel('generation')
    plt.ylabel('average-fitness')
    plt.title('Average Fitness per Generation')
    plt.legend()
    plt.show()
    '''



plots()






#EA(lm,gen_size,probability,default_fitness)
