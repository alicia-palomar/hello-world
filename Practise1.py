import numpy as np
import matplotlib.pyplot as plt

patient_data = np.load("C:\Alicia\projects\hello-world\PatientData.npy")

#plot some slices
for k in [0, 50, 100, 150]:
    plt.figure()
    plt.imshow(patient_data[k], cmap=plt.cm.bone)

#Histogram
hist, bins = np.histogram(patient_data[50].flatten(), 256, [0, 256])
plt.hist(patient_data[50].flatter(), 256, [0, 256])
plt.show()



