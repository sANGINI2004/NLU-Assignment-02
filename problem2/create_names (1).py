
import random

# some common indian sounding starting parts
prefix = ["ra","ro","ri","su","sa","an","ar","vi","ni","ka",
"ma","mo","de","di","shi","sha","pra","pre","tri","ga",
"ha","ja","ku","la","pa","ta","ya"]

# ending parts of names
suffix = ["ul","esh","it","an","av","ay","ant","endra","deep","ansh",
"ika","ita","ali","ini","usha","ish","raj",
"dev","pal","jeet","nath","veer","preet"]

names = []

# generate 1000 names randomly
for i in range(1000):
    
    # combine prefix and suffix
    name = random.choice(prefix) + random.choice(suffix)
    
    # make first letter capital
    name = name.capitalize()
    
    names.append(name)

# save names to file
with open("TrainingNames.txt","w") as f:
    for n in names:
        f.write(n+"\n")

print("dataset created with", len(names), "names")
