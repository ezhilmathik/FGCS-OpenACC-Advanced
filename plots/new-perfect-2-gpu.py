import numpy as np 
import matplotlib.pyplot as plt 

barWidth = 0.075

SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# set width of bar 
fig = plt.subplots(figsize =(12, 8)) 
xx=[512, 896, 1280] 
# set height of bar

#OpenACC_Version1 = [ 4.490563,  15.626899, 40.043076]
#OpenACC_Version2 = [ 3.037851,  13.411582, 35.163471]

CUDA_Version1 = [4.409136, 17.638935, 46.965247]
OpenACC_Version3 = [ 9.648185,  33.983631,  87.009237]

CUDA_Version2 = [4.214998, 16.64712, 45.555308]
OpenACC_Version4 = [ 7.711648,  26.231482, 67.50802]

CUDA_Version3 = [3.553327, 14.614045, 40.348518]
OpenACC_Version5 = [4.356552,  18.52197, 51.05624]

CUDA_Version4 = [2.148756,  7.338788,  19.01059]
OpenACC_Version6 = [3.395404, 13.605388, 36.945137]

# Set position of bar on X axis 
br1 = np.arange(len(xx))
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5] 
br7 = [x + barWidth for x in br6]
br8 = [x + barWidth for x in br7] 


# Make the plot
plt.bar(br1, CUDA_Version1, color ='b', linewidth=0.01, width=barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-1') 
plt.bar(br2, OpenACC_Version3, color ='g', linewidth=0.01, width=barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-1')
plt.bar(br3, CUDA_Version2, color ='r', linewidth=0.01, width=barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-2') 
plt.bar(br4, OpenACC_Version4, color ='c', linewidth=0.01, width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-2')
plt.bar(br5, CUDA_Version3, color ='m', linewidth=0.01, width = barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-3') 
plt.bar(br6, OpenACC_Version5, color ='y', linewidth=0.01, width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-3')
plt.bar(br7, CUDA_Version4, color ='k', linewidth=0.01, width = barWidth, 
        edgecolor ='grey', label ='CUDA Multi-GPU Version-4') 
plt.bar(br8, OpenACC_Version6, color ='tab:pink', linewidth=0.01, width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-4') 


# Adding Xticks 
plt.xlabel('Grid Size', fontsize = MEDIUM_SIZE)#, fontweight ='bold', fontsize = 15) 
plt.ylabel('Time in Seconds', fontsize = MEDIUM_SIZE)#, fontweight ='bold', fontsize = 15) 
#plt.xticks([r + barWidth for r in range(len(xx))], 
#        ['128', '256', '384', '512', '640'])
plt.xticks(br1 + 2.*barWidth, xx)

plt.minorticks_on()
# Customize the major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.2)
# Customize the minor grid
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.2)
plt.ylim(0, 120)
#plt.grid()
#plt.legend()
plt.legend(ncol=2, loc='upper left')
plt.title("2 GPUs")
plt.savefig("new-perfect-2-gpu.pdf", format="pdf", bbox_inches="tight")
plt.show() 
