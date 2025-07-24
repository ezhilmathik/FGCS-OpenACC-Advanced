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

CUDA_Version1 = [0.936498,	0.810796,	0.476589,	0.48461,	0.382945,	0.403748,	0.452909,	0.461041]
OpenACC_Version3 = [1.414407,	1.290756,	1.293203,	1.312567,	0.927174,	0.965737,	1.210505,	1.233836]

CUDA_Version2 = [0.943755,	0.818649,	0.451115,	0.449828,	0.340966,	0.352596,	0.426852,	0.431237]
OpenACC_Version4 = [1.41357,	1.288064,	1.004182,	0.930436,	0.649392,	0.677155,	0.874117,	0.874155]

CUDA_Version3 = [0.759406,	0.674369,	0.445071,	0.384746,	0.279194,	0.284778,	0.36175,	0.364013]
OpenACC_Version5 = [0.98211,	0.898157,	0.510296,	0.457102,	0.322133,	0.329607,	0.429799,	0.427348]

CUDA_Version4 = [0.037141,	0.036979,	0.435414,	0.377691,	0.269186,	0.282009,	0.352506,	0.35783]
OpenACC_Version6 = [0.461707,	0.461234,	0.500709,	0.450594,	0.327308,	0.334288,	0.428781,	0.430783]

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
plt.title("2 GPUs; Grid size=512")
plt.legend(ncol=2, loc='upper right')
plt.xticks(rotation=10)
#plt.ylim(0, 3)
plt.savefig("new-512-data-2-gpu.pdf", format="pdf", bbox_inches="tight")
#plt.ylim(-1, 1)
plt.show() 
