
import torch

# read raw dataset
with open("TrainingNames.txt") as f:
    names = f.read().splitlines()

print("total names:", len(names))

# remove duplicates
names = list(set(names))

print("unique names:", len(names))

# save unique training names (needed for evaluation later)
with open("train_unique.txt","w") as f:
    for n in names:
        f.write(n + "\n")

print("saved train_unique.txt")

# build vocabulary
chars = sorted(list(set("".join(names))))
print("vocab size:", len(chars))

stoi = {ch:i for i,ch in enumerate(chars)}

inputs = []
targets = []

# create character transition pairs
for name in names:
    for i in range(len(name)-1):
        inputs.append(stoi[name[i]])
        targets.append(stoi[name[i+1]])

X = torch.tensor(inputs)
Y = torch.tensor(targets)

print("training pairs:", X.shape)
