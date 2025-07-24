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

CUDA_Version1 = [1.422752,	1.232435,	0.741997,	0.746862,	0.688389,	0.71281,	0.7388,	 0.737595]
OpenACC_Version3 = [1.860563,   1.671995,       2.158413,       2.171481,       1.743873,       1.785507,       2.077351, 2.09609]

CUDA_Version2 = [1.41025,	1.224471,	0.585755,	0.589796,	0.543282,	0.549168,	0.57625,	0.578271]
OpenACC_Version4 = [1.858843, 1.673091, 2.170537, 1.918951, 1.499579, 1.535659, 1.834742, 1.8689]

CUDA_Version3 = [0.90897,	0.836981,	0.462942,	0.38542,	0.326095,	0.332467,	0.372304,	0.38082]
OpenACC_Version5 = [1.030787,	0.956506,	0.633767,	0.500457,	0.422192,	0.426679,	0.47091,	0.484174]

CUDA_Version4 = [0.014674,	0.01238,	0.422823,	0.345674,	0.280907,	0.2894,	0.331293,	0.345018]
OpenACC_Version6 = [0.431525,	0.427483,	0.649532,	0.560455,	0.42485,	0.43026,	0.478525,	0.484571]

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
        edgecolor ='grey', label ='CUDA Multi-GPU Version-2') 
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
plt.title("4 GPUs; Grid size=512")
plt.legend(ncol=2, loc='upper right')
plt.xticks(rotation=10)
#plt.ylim(0, 3)
plt.savefig("new-512-data-4-gpu.pdf", format="pdf", bbox_inches="tight")
#plt.ylim(-1, 1)
plt.show() 
