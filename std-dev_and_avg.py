'''
This is a simple function/code that calculates the average and std. dev
for n trails of running the GA/CGA
'''
#DRIVER
def dev_and_avg(num_trails):
    #stats_array to store the array of arrays used to calculate the average and standard deviation
    stats_array_avgavg=[]
    stats_array_minavg=[]
    deviation_min=[]
    deviation_avg=[]

    # run the EA according to the number of trails
    # and store each min and average array in stats_array
    for _ in range(num_trails):
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
