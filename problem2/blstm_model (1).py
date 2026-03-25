
import torch
import torch.nn as nn
import torch.optim as optim
import random

# load unique training names
with open("train_unique.txt") as f:
    names = f.read().splitlines()

# build vocabulary
chars = sorted(list(set("".join(names))))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

# hyperparameters
hidden_size = 64
learning_rate = 0.01
epochs = 20

# BLSTM model
class BLSTMModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # embedding layer converts index -> dense vector
        self.embed = nn.Embedding(vocab_size, hidden_size)
        
        # bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # output layer
        self.fc = nn.Linear(hidden_size*2, vocab_size)
    
    def forward(self, x):
        
        x = self.embed(x)
        out,_ = self.lstm(x)
        out = self.fc(out)
        
        return out

model = BLSTMModel()

# parameter count (important for report)
total_params = sum(p.numel() for p in model.parameters())
print("Trainable Parameters:", total_params)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# training loop
for epoch in range(epochs):
    
    total_loss = 0
    
    for name in names:
        
        # convert name to indices
        seq = torch.tensor([stoi[c] for c in name])
        
        inp = seq[:-1].unsqueeze(0)
        target = seq[1:]
        
        optimizer.zero_grad()
        
        output = model(inp)
        
        loss = loss_fn(output.squeeze(0), target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print("epoch", epoch, "loss", total_loss)


# name generation
def generate_name(start='R', max_len=10):
    
    model.eval()
    
    name = start
    
    for i in range(max_len):
        
        x = torch.tensor([[stoi[name[-1]]]])
        
        out = model(x)
        
        probs = torch.softmax(out[0,-1], dim=0)
        
        idx = torch.multinomial(probs,1).item()
        
        name += itos[idx]
    
    return name


# save generated samples
with open("blstm_samples.txt","w") as f:
    for _ in range(200):
        f.write(generate_name(random.choice(chars)) + "\n")

print("generated samples saved")
