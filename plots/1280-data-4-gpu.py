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
#xx=['Data Partition 1', 'Data Partition 2', 'X-dir-1', 'X-dir-2', 'Y-dir-1', 'Y-dir-2', 'Z-dir-1', 'Z-dir-2']
xx=['Data Partition 1', 'Data Partition 2', 'X-forward.', 'X-backward', 'Y-forward', 'Y-backward', 'Z-forward', 'Z-backward']
# set height of bar

OpenACC_Version3 = [29.133469,	26.235898,	11.146771,	13.205089,	8.970114,	9.618041,	10.497581,	12.224698]
OpenACC_Version4 = [29.064156,	26.191545,	5.641779,	5.419882,	4.194472,	4.361156,	5.199993,	5.040827]
OpenACC_Version5 = [15.074581,	14.200599,	2.381233,	2.266015,	1.691852,	1.790855,	2.033578,	2.07162]
OpenACC_Version6 = [6.948468,	6.796933,	2.36328,	2.267015,	1.694889,	1.768,	        2.034438,	2.073027]


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

#plt.bar(br5, OpenACC_Version5, color ='m', width = barWidth, 
#        edgecolor ='grey', label ='OpenACC_Version5') 
#plt.bar(br6, OpenACC_Version6, color ='y', width = barWidth, 
#        edgecolor ='grey', label ='OpenACC_Version6') 

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
plt.title("4 GPUs; Grid size=1280")
plt.savefig("1280-data-4-gpu.pdf", format="pdf", bbox_inches="tight")
plt.show() 
