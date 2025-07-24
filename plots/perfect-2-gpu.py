import numpy as np 
import matplotlib.pyplot as plt 

SMALL_SIZE = 8
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# set width of bar 
barWidth = 0.125
fig = plt.subplots(figsize =(12, 8)) 
xx=[128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280] 
# set height of bar

OpenACC_Version1 = [1.322513, 1.763118, 2.760048, 4.490563, 7.241588, 10.509596, 15.626899, 21.987092, 29.976988, 40.043076]
OpenACC_Version2 = [0.182216, 0.607416, 1.502711, 3.037851, 5.495726, 8.727377, 13.411582, 19.195663, 26.060296, 35.163471]
OpenACC_Version3 = [1.085306, 2.611826, 5.596105, 9.872706, 14.963538, 22.814888, 34.27857, 47.731985, 65.696904, 87.009237]
OpenACC_Version4 = [0.946321, 2.226013, 4.70315, 7.711648, 12.468318, 17.689174, 26.232939, 36.913141, 51.90217, 67.511725]
OpenACC_Version5 = [0.417384, 1.040952, 2.293754, 4.356552, 7.295485, 11.860404, 18.52197, 26.939286, 37.217255, 51.05624]
OpenACC_Version6 = [0.397951, 0.891205, 1.894382, 3.395404, 5.446152, 8.691786, 13.259332, 19.535617, 26.371056, 35.835238]

# Set position of bar on X axis 
br1 = np.arange(len(xx))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4] 
br6 = [x + barWidth for x in br5]
 
# Make the plot
plt.bar(br1, OpenACC_Version1, color ='r', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Single-GPU Version-1') 
plt.bar(br2, OpenACC_Version2, color ='b', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Single-GPU Version-2') 
plt.bar(br3, OpenACC_Version3, color ='g', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-1') 
plt.bar(br4, OpenACC_Version4, color ='c', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-2') 
plt.bar(br5, OpenACC_Version5, color ='m', width = barWidth, 
        edgecolor ='grey', label ='OpenACC Multi-GPU Version-3') 
plt.bar(br6, OpenACC_Version6, color ='y', width = barWidth, 
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
plt.ylim(0, 100)
#plt.grid()
#plt.legend()
plt.legend(ncol=2, loc='upper center')
plt.title("2 GPUs")
plt.savefig("perfect-2-gpu.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show() 
