import numpy as np 
import matplotlib.pyplot as plt 

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# set width of bar 
barWidth = 0.1
fig = plt.subplots(figsize =(12, 8)) 
#xx=['Data Partition 1', 'Data Partition 2', 'X-dir-1', 'X-dir-2', 'Y-dir-1', 'Y-dir-2', 'Z-dir-1', 'Z-dir-2']
xx=['Data Partition 1', 'Data Partition 2', 'X-forward.', 'X-backward', 'Y-forward', 'Y-backward', 'Z-forward', 'Z-backward']
# set height of bar

CUDA_Version1 = [5.027701,	4.381841,	1.553325,	1.544498,	1.120023,	1.178488,	1.406557,	1.426502]
OpenACC_Version3 = [7.533962,	6.858092,	3.721768,	3.787879,	2.464812,	2.618707,	3.457506,	3.540905]

CUDA_Version2 = [5.013657,	4.359051,	1.407573,	1.37686,	1.008875,	1.027875,	1.233352,	1.219877]
OpenACC_Version4 = [7.490268,	6.824818,	2.268458,	2.255761,	1.607887,	1.623092,	2.050542,	2.110656] 

CUDA_Version3 = [4.00811,	3.566082,	1.43362,	1.336479,	0.920706,	0.95573,	1.197098,	1.19622]
OpenACC_Version5 = [5.301681,	4.79531,	1.748059,	1.619201,	1.048303,	1.091424,	1.437838,	1.480154]

CUDA_Version4 = [0.194572,	0.194089,	1.379753,	1.331312,	0.921302,	0.957767,	1.162333,	1.19766]
OpenACC_Version6 = [2.49113,	2.428507,	1.674538,	1.618146,	1.045174,	1.089081,	1.435399,	1.477357]

# Set position of bar on X axis 
br1 = np.arange(len(xx)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]
br7 = [x + barWidth for x in br6]
br8 = [x + barWidth for x in br7] 

#br5 = [x + barWidth for x in br4]
#br6 = [x + barWidth for x in br5] 
 
# Make the plot
plt.bar(br1, CUDA_Version1, color ='b', width = barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-1') 
plt.bar(br2, OpenACC_Version3, color ='g', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-1')

plt.bar(br3, CUDA_Version2, color ='r', width = barWidth, 
        edgecolor ='grey', label ='CUD Multi-GPU Version-2') 
plt.bar(br4, OpenACC_Version4, color ='c', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-2')

plt.bar(br5, CUDA_Version3, color ='m', width = barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-3') 
plt.bar(br6, OpenACC_Version5, color ='y', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-3')

plt.bar(br7, CUDA_Version4, color ='k', width = barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-4') 
plt.bar(br8, OpenACC_Version6, color ='w', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-4') 

#plt.bar(br5, OpenACC_Version5, color ='m', width = barWidth, 
#        edgecolor ='grey', label ='OpenACC_Version5') 
#plt.bar(br6, OpenACC_Version6, color ='y', width = barWidth, 
#        edgecolor ='grey', label ='OpenACC_Version6') 

# Adding Xticks 
#plt.xlabel('Domain Size', fontsize = MEDIUM_SIZE)#, fontweight ='bold', fontsize = 15) 
plt.ylabel('Time in Seconds', fontsize = MEDIUM_SIZE)#, fontweight ='bold', fontsize = 15) 
#plt.xticks([r + barWidth for r in range(len(xx))], 
#        ['128', '256', '384', '512', '640'])
plt.xticks(br1 + 2*barWidth, xx)

plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5')#, color='red')
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5')#, color='black')

#plt.grid()
#plt.legend()
plt.title("2 GPUs; Grid size=896")
plt.legend(ncol=2, loc='upper right')
plt.xticks(rotation=10)
#plt.ylim(0, 3)
plt.savefig("new-896-data-2-gpu.pdf", format="pdf", bbox_inches="tight")
#plt.ylim(-1, 1)
plt.show() 
