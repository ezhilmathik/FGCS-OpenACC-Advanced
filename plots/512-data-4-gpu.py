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
OpenACC_Version3 = [1.860563,	1.671995,	2.158413,	2.171481,	1.743873,	1.785507,	2.077351,	2.09609]
OpenACC_Version4 = [1.858843,	1.673091,	2.170537,	1.918951,	1.499579,	1.535659,	1.834742,	1.8689]
OpenACC_Version5 = [1.030787,	0.956506,	0.633767,	0.500457,	0.422192,	0.426679,	0.47091,	0.484174]
OpenACC_Version6 = [0.431525,	0.427483,	0.649532,	0.560455,	0.42485,	0.43026,	0.478525,	0.484571]
    
# Set position of bar on X axis 
br1 = np.arange(len(xx)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3]

#br5 = [x + barWidth for x in br4]
#br6 = [x + barWidth for x in br5] 
 
# Make the plot
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
plt.xticks(br1 + 2*barWidth, xx)

plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.15')#, color='red')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.25')#, color='black')

#plt.grid()
#plt.legend()
plt.legend(ncol=2, loc='upper center')
plt.xticks(rotation=10)
plt.title("4 GPUs; Grid size=512")
plt.savefig("512-data-4-gpu.pdf", format="pdf", bbox_inches="tight")
plt.show() 
