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

OpenACC_Version3 = [21.732845,	19.750129,	8.644865,	8.924055,	5.543976,	6.021993,	8.128753,	8.341512]
OpenACC_Version4 = [21.736435,	19.808047,	4.909474,	4.971615,	3.302347,	3.400331,	4.635277,	4.744494]
OpenACC_Version5 = [15.318181,	14.074792,	4.298609,	4.201413,	2.633307,	2.849818,	3.762267,	3.917853]
OpenACC_Version6 = [7.348537,	7.139836,	4.16565,	4.220554,	2.613473,	2.761778,	3.738871,	3.846539]


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
plt.title("2 GPUs; Grid size=1280")
plt.savefig("1280-data-2-gpu.pdf", format="pdf", bbox_inches="tight")
plt.show() 
