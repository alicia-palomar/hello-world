import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import pyplot, cm

patient_folder = "C:/Users/apa10431/Documents/CMCiB/AdiposeTissueSegmentation/JRR_axial/"

list_dicom_names = []
for i in range(num_slices):
    list_dicom_names.append(os.listdir(patient_folder)[i])

slices = [pydicom.dcmread(patient_folder + "/" + s) for s in os.listdir(patient_folder)]

num_slices = np.shape(os.listdir(patient_folder))[0]

Slice_locations = []
for i in range(num_slices):
    Slice_locations.append(slices[i].SliceLocation)

#That slice with minimum SliceLocation (more  negative value) is the 1st slice

sorted_Slice_locations=sorted(Slice_locations)
idx_Slice=[]
for i in range(num_slices):
    for j in range(num_slices):
        if sorted_Slice_locations[i] == Slice_locations[j]:
            idx_Slice.append(j)

# In idx_Slice we find the index of the slice based on the previous sorting process

patient_data = []
for i in range(num_slices):
    patient_data.append(slices[idx_Slice[i]].pixel_array)

for k in [5, 25, 50, 100, 150]:
    plt.figure()
    plt.imshow(patient_data[k], cmap=plt.cm.bone)




##########
    plt.figure()
    plt.imshow(slices[i].pixel_array, cmap=plt.cm.bone)







patient_data = []
for s in patient_folder:  # s for each slice
    patient_data.append(s.pixel_array)





filename="C:/Users/apa10431/Documents/CMCiB/AdiposeTissueSegmentation/000004-01/1.2.392.200036.9116.4.2.8054.5591.20180124113242542.3.2738.dcm"
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)



#def load_scan(path):
#     slices = [pydicom.dcmread(path + "/" + s) for s in
#               os.listdir(path)]
#     slices = [s for s in slices if "SliceLocation" in s]
#     slices.sort(key = lambda x: int(x.InstanceNumber))
#     try:
#         slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
#     except:
#         slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
#     for s in slices:
#         s.SliceThickness = slice_thickness
#     return slices