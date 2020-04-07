import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import pyplot, cm

patient_folder = "C:/Users/apa10431/Documents/CMCiB/AdiposeTissueSegmentation/JRR_axial/"
num_slices = np.shape(os.listdir(patient_folder))[0]

list_dicom_names = []
for i in range(num_slices):
    list_dicom_names.append(os.listdir(patient_folder)[i])

np.save("ListDicomNames", list_dicom_names)

slices = [pydicom.dcmread(patient_folder + "/" + s) for s in os.listdir(patient_folder)]

Slice_locations = []
for i in range(num_slices):
    Slice_locations.append(slices[i].SliceLocation)

#That slice with minimum SliceLocation (more  negative value) is the last slice

sorted_Slice_locations = sorted(Slice_locations, reverse=True) # reverse set at true for descending order
np.save("sorted_Slice_locations", sorted_Slice_locations)
idx_Slice=[]
for i in range(num_slices):
    for j in range(num_slices):
        if sorted_Slice_locations[i] == Slice_locations[j]:
            idx_Slice.append(j)

# In idx_Slice we find the index of the slice based on the previous sorting process
np.save("SliceIndexes", idx_Slice)

patient_data = []
for i in range(num_slices):
    patient_data.append(slices[idx_Slice[i]].pixel_array)

np.save("PatientData", patient_data)

