from BerkleyDataSet import BerkeleyDataset
from PressleyModel import PressleyModel
import torch
from torch.nn import functional as F

# define hyperparameters
lr = 0.0003 # learning rate
bs = 10 # batch size
epochs = 100 # number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device
nmbed = 256
seq_len = 5

eval_interval = 300

# create model instance and move to device
model = PressleyModel(nmbed)
model.to(device)

# create optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_dataset = BerkeleyDataset(seq_len, 0.3, True)
test_dataset = BerkeleyDataset(seq_len, 0.3, False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=4)

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
        loss = F.cross_entropy(output, next_action.to(device))
        
        # print or log loss and other metrics as needed
        print(f'Epoch {epoch}, Loss: {loss.item()}')

        # backpropagate loss
        loss.backward()

        # update model parameters
        optimizer.step()

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

torch.save(model.state_dict(), "pressley.pth")