np.load()


for k in [0, 5, 25, 50, 100, 150]:
    plt.figure()
    plt.imshow(patient_data[k], cmap=plt.cm.bone)