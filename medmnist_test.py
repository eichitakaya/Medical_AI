from MedMNIST import MedMNIST

train_dataset = MedMNIST("adrenal3d", traintestval="train")

print(len(train_dataset))
print(train_dataset.data.shape)
print(train_dataset.targets[1][0])