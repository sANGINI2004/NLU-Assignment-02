
# evaluation script for name generation models

# load training unique names
with open("train_unique.txt") as f:
    train = set(f.read().splitlines())

# load generated names
with open("attention_samples.txt") as f:
    gen = f.read().splitlines()

total = len(gen)

# novelty calculation
novel = 0
for name in gen:
    if name not in train:
        novel += 1

novelty_rate = novel / total

# diversity calculation
unique_generated = len(set(gen))
diversity = unique_generated / total

print("Total Generated:", total)
print("Novel Names:", novel)
print("Novelty Rate:", novelty_rate)
print("Diversity:", diversity)
