import numpy as np 
import matplotlib.pyplot as plt 

SMALL_SIZE = 8
MEDIUM_SIZE = 14
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# set width of bar 
barWidth = 0.15
fig = plt.subplots(figsize =(12, 8)) 
xx=['Data Partition 1', 'Data Partition 2', 'X-forward.', 'X-backward', 'Y-forward', 'Y-backward', 'Z-forward', 'Z-backward']
# set height of bar

OpenACC_Version3 = [7.533962,	6.858092,	3.721768,	3.787879,	2.464812,	2.618707,	3.457506,	3.540905]
OpenACC_Version4 = [7.490268,	6.824818,	2.268458,	2.255761,	1.607887,	1.623092,	2.050542,	2.110656]
OpenACC_Version5 = [5.301681,	4.79531,	1.748059,	1.619201,	1.048303,	1.091424,	1.437838,	1.480154]                    
OpenACC_Version6 = [2.624546,	2.567673,	1.725525,	1.620275,	1.05345,	1.097071,	1.437928,	1.47892]
                   

# Set position of bar on X axis 
br1 = np.arange(len(xx)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
 
plt.bar(br1, OpenACC_Version3, color ='b', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-1') 
plt.bar(br2, OpenACC_Version4, color ='g', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-2') 
plt.bar(br3, OpenACC_Version5, color ='r', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-3') 
plt.bar(br4, OpenACC_Version6, color ='y', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-4') 


# Adding Xticks 
#plt.xlabel('Domain Size', fontsize = 12)#, fontweight ='bold', fontsize = 15) 
plt.ylabel('Time in Seconds', fontsize = 14)#, fontweight ='bold', fontsize = 15) 
#plt.xticks([r + barWidth for r in range(len(xx))], 
#        ['128', '256', '384', '512', '640'])
plt.xticks(br1 + 2.*barWidth, xx)

plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.15')#, color='red')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.25')#, color='black')

#plt.grid()
#plt.legend()
plt.legend(ncol=2, loc='upper center')
plt.xticks(rotation=10)
plt.title("2 GPUs; Grid size=896")
plt.savefig("896-data-2-gpu.pdf", format="pdf", bbox_inches="tight")
plt.show() 
