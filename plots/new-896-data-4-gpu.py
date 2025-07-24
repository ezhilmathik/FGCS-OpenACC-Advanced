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

CUDA_Version1 = [7.474661,	6.487997,	1.605538,	1.581814,	1.357883,	1.422088,	1.542313,	1.546131]
OpenACC_Version3 = [10.06534,	9.08055,	6.166066,	6.314743,	4.412236,	4.636575,	5.824098,	5.977227]

CUDA_Version2 = [7.475159,	6.50143,	1.288263,	1.286139,	1.067091,	1.126767,	1.238002,	1.222547]
OpenACC_Version4 = [10.115752,	9.113517,	4.845618,	4.678547,	3.341074,	3.574069,	4.50448,	4.494232]

CUDA_Version3 = [4.643624,	4.351524,	1.008599,	0.902743,	0.697877,	0.730827,	0.845265,	0.830756]
OpenACC_Version5 = [5.263997,	4.971469,	1.310915,	1.142033,	0.876397,	0.928961,	1.08377,	1.098201]

CUDA_Version4 = [0.070205,	0.061026,	0.98275,	0.898205,	0.658897,	0.689721,	0.814912,	0.811961]
OpenACC_Version6 = [2.326263,	2.329914,	1.308522,	1.151457,	0.875157,	0.914312,	1.098876,	1.091788]

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
plt.title("4 GPUs; Grid size=896")
plt.legend(ncol=2, loc='upper right')
plt.xticks(rotation=10)
#plt.ylim(0, 3)
plt.savefig("new-896-data-4-gpu.pdf", format="pdf", bbox_inches="tight")
#plt.ylim(-1, 1)
plt.show() 
