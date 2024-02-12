from BerkleyDataSet import BerkeleyDataset
from PressleyModel import PressleyModel
import torch
from torch.nn import functional as F
from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt


@dataclass
class Args:
    batch_size: int = 10
    train_folder: str = "/mnt/d/Downloads-D/scripted_6_18/scripted_raw/sweep_12-03/"

args = tyro.cli(Args)

# define hyperparameters
lr = 0.00001 # learning rate
bs = args.batch_size # batch size
epochs = 100 # number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device
nmbed = 256
seq_len = 5

eval_interval = 300

# create model instance and move to device
model = PressleyModel(nmbed, seq_len)
model.to(device)

# create optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_dataset = BerkeleyDataset(seq_len, args.train_folder, 0.3, True)
test_dataset = BerkeleyDataset(seq_len, args.train_folder, 0.3, False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=4)

train_losses = torch.zeros(int(epochs * len(test_dataloader) / bs))
k = 0

print('Training starts')
for epoch in range(epochs):
    for batch in train_dataloader:
        # extract images and actions from batch 
        images = batch['image']
        actions = batch['action']

        # set gradients to zero
        optimizer.zero_grad()

        # pass images and actions to model and get output actions (past seq_len samples)
        output = model(images, actions)

        # get next action
        next_action = batch['next_action']
        
        # compute loss
        # Do i need softmax if I am using an action decoder?
        loss = F.cross_entropy(output, next_action.to(device))
        
        # print or log loss and other metrics as needed
        # print(f'Epoch {epoch}, Loss: {loss.item()}')
        train_losses[k] = loss.item()
        k = k + 1
        # backpropagate loss
        loss.backward()

        # update model parameters
        optimizer.step()
    print(f'Epoch {epoch}, Mean Loss till now: {train_losses.mean()}')

plt.plot(train_losses)
plt.xlabel('Iter')
plt.ylabel('Train Loss')
plt.title('Train Losses')
plt.show()


# validation
model.eval()
losses = torch.zeros(len(test_dataloader) / bs)
k = 0
for batch in test_dataloader:
    images = batch['image']
    actions = batch['action']
    next_action = batch['next_action']
    output = model(images, actions)
    
    # compute loss
    loss = F.cross_entropy(output, next_action.to(device))
    losses[k] = loss.item()
    k = k + 1

print(f'Validation Loss: {losses.mean()}')

plt.plot(losses)
plt.xlabel('Iter')
plt.ylabel('Validation Loss')
plt.title('Validation Losses')
plt.show()

torch.save(model.state_dict(), "pressley.pth")