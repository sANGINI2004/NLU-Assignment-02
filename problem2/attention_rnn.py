
import torch
import torch.nn as nn
import torch.optim as optim
import random

# load training names
with open("train_unique.txt") as f:
    raw_names = f.read().splitlines()

# add start and end tokens
names = ["<S>" + n + "<E>" for n in raw_names]

# build vocab
chars = sorted(list(set("".join(names))))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

hidden_size = 64
epochs = 20
lr = 0.01
temperature = 0.8


class AttentionRNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

        # attention scoring
        self.attn = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, x):

        x = self.embed(x)

        outputs, _ = self.rnn(x)

        # attention weights
        scores = torch.tanh(self.attn(outputs))
        weights = torch.softmax(scores, dim=1)

        context = torch.sum(weights * outputs, dim=1)

        last = outputs[:, -1, :]

        combined = torch.cat([context, last], dim=1)

        out = self.fc(combined)

        return out


model = AttentionRNN()

params = sum(p.numel() for p in model.parameters())
print("Trainable Parameters:", params)

opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


# training loop (next char prediction at each step)
for ep in range(epochs):

    total_loss = 0

    for name in names:

        seq = torch.tensor([stoi[c] for c in name])

        for i in range(len(seq)-1):

            inp = seq[:i+1].unsqueeze(0)
            target = seq[i+1].unsqueeze(0)

            opt.zero_grad()

            out = model(inp)

            loss = loss_fn(out, target)

            loss.backward()
            opt.step()

            total_loss += loss.item()

    print("epoch", ep, "loss", total_loss)


# generation using full history
def generate_name():

    model.eval()

    name = "<S>"

    for _ in range(12):

        seq = torch.tensor([[stoi[c] for c in name]])

        out = model(seq)

        probs = torch.softmax(out[0]/temperature, dim=0)

        idx = torch.multinomial(probs,1).item()

        ch = itos[idx]

        if ch == "<E>":
            break

        name += ch

    return name.replace("<S>","")


# save samples
with open("attention_samples.txt","w") as f:
    for _ in range(200):
        f.write(generate_name() + "\n")

print("new attention samples saved")
