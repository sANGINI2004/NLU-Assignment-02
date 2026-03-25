

import torch
import torch.nn as nn
import torch.optim as optim
import random

# load dataset
with open("TrainingNames.txt") as f:
    names = f.read().splitlines()

# remove duplicates
names = list(set(names))

# build vocab
chars = sorted(list(set("".join(names))))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

# hyperparameters
hidden_size = 64
learning_rate = 0.01
epochs = 20

# vanilla rnn model from scratch
class VanillaRNN(nn.Module):
    
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # weight matrices
        self.Wxh = nn.Parameter(torch.randn(vocab_size, hidden_size)*0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)
        self.Why = nn.Parameter(torch.randn(hidden_size, vocab_size)*0.01)
        
        # biases
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(vocab_size))
        
    def forward(self, inputs):
        
        h = torch.zeros(self.hidden_size)
        loss = 0
        
        for i in range(len(inputs)-1):
            
            x = torch.zeros(vocab_size)
            x[stoi[inputs[i]]] = 1   # one hot
            
            # hidden state update
            h = torch.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
            
            # output logits
            y = h @ self.Why + self.by
            
            target = torch.tensor([stoi[inputs[i+1]]])
            
            loss += nn.functional.cross_entropy(y.view(1,-1), target)
        
        return loss


model = VanillaRNN(vocab_size, hidden_size)


# count total trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print("Trainable Parameters:", total_params)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(epochs):
    
    total_loss = 0
    
    for name in names:
        
        optimizer.zero_grad()
        
        loss = model(name)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    
    print("epoch", epoch, "loss", total_loss)


# function to generate name
def generate_name(start_char='R', max_len=10):
    
    h = torch.zeros(hidden_size)
    name = start_char
    
    for i in range(max_len):
        
        x = torch.zeros(vocab_size)
        x[stoi[name[-1]]] = 1
        
        h = torch.tanh(x @ model.Wxh + h @ model.Whh + model.bh)
        y = h @ model.Why + model.by
        
        probs = torch.softmax(y, dim=0)
        
        idx = torch.multinomial(probs, 1).item()
        
        next_char = itos[idx]
        name += next_char
    
    return name


print("\nGenerated Names:")
for _ in range(10):
    print(generate_name(random.choice(chars)))



# save generated samples
with open("vanilla_rnn_samples.txt","w") as f:
    for _ in range(200):
        f.write(generate_name(random.choice(chars))+"\n")
