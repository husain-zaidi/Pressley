from BerkleyDataSet import BerkeleyDataset
from PressleyModel import PressleyModel
import torch
from torch.nn import functional as F

# define hyperparameters
lr = 0.0003 # learning rate
bs = 10 # batch size
epochs = 100 # number of epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device

# create model instance and move to device
model = PressleyModel()
model.to(device)

# create optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = BerkeleyDataset()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4)

for epoch in range(epochs):
    for batch in dataloader:
        # extract images and actions from batch 
        images = batch['image']
        actions = batch['action']

        # set gradients to zero
        optimizer.zero_grad()

        # pass images and actions to model and get output actions (past 5 samples)
        output = model(images, actions, training=True)

        # compute loss
        loss = F.cross_entropy(output, actions.to(device))

        # backpropagate loss
        loss.backward()

        # update model parameters
        optimizer.step()

        # print or log loss and other metrics as needed
        print(f'Epoch {epoch}, Loss: {loss.item()}')

torch.save(model.state_dict(), "pressley.pth")