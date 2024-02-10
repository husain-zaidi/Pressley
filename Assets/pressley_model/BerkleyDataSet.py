import torch
import os
import pickle
import glob
import numpy as np
from PIL import Image
import torchvision

data_path = "/mnt/d/Downloads-D/scripted_6_18/scripted_raw/sweep_12-03/"
# 2022-12-*05_13-16-57* -> raw"-> "traj_group*"-> "traj* -> policy_out.pkl, images0 -> im_*.jpg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device

class BerkeleyDataset(torch.utils.data.Dataset):

    def __init__(self, seq_len, test_train_split=0.3, train=True):
        search_path = os.path.join(data_path,  "2022-12-*", "raw", "traj_group*", "traj*")
        all_traj = glob.glob(search_path)
        self.image_filenames = np.array([])
        self.actions = np.empty((0,7))

        self.seq_len = seq_len

        n = int((1-test_train_split)*len(all_traj))
        train_data = all_traj[:n]
        val_data = all_traj[n:]
        all_traj = train_data if train else val_data
        for folder in all_traj:
            # get all images
            images_files = glob.glob(os.path.join(folder , "images0", "im_*.jpg"))
            self.image_filenames = np.append(self.image_filenames, images_files[1:])
            action = self.load_actions(folder)
            action_np = np.stack(action, axis=0)
            self.actions =  np.append(self.actions, action_np, axis=0)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)), # resize images to match model input size
            torchvision.transforms.ToTensor(), # convert images to tensors
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalize images using ImageNet mean and std
        ])

    def __getitem__(self, idx):
        item = {}
        # load self.seq_len prev images and corresponding action
        images = torch.tensor([])
        for i in range(self.seq_len):
            image =  Image.open(self.image_filenames[idx - i])
            image = self.transforms(image)
            image = torch.unsqueeze(image, dim=0) # add extra dim for catenating images
            images = torch.cat((image, images), dim=0)
        item['image'] = images

        actions = torch.tensor([], dtype=torch.float32)
        for i in range(self.seq_len):
            action = torch.tensor(self.actions[idx- i], dtype=torch.float32)
            action = torch.unsqueeze(action, dim=0)
            actions =  torch.cat((action, actions), dim=0)
        item['action'] = actions

        next_actions = torch.tensor([], dtype=torch.float32)
        for i in range(self.seq_len):
            next_action = torch.tensor(self.actions[idx+1 - i], dtype=torch.float32)
            next_action = torch.unsqueeze(next_action, dim=0)
            next_actions =  torch.cat((next_action, next_actions), dim=0)
        item['next_action'] = next_actions

        return item
    
    def load_actions(self, path):  # gets actions
        fp = os.path.join(path, "policy_out.pkl")
        with open(fp, "rb") as f:
            act_list = pickle.load(f)
        if isinstance(act_list[0], dict):
            act_list = [x["actions"] for x in act_list]
        return act_list #arrays of 7 elements each
    
    def __len__(self):
        return len(self.image_filenames) - self.seq_len 