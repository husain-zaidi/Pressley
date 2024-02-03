import glob
import logging
import os
import pickle
import random
import tqdm

path = "/mnt/d/Downloads-D/scripted_6_18/scripted_raw/sweep_12-03/2022-12-05_13-16-57"
# path = "/mnt/d/Downloads-D/scripted_6_18/scripted_raw/sweep_12-03/2022-12-05_13-16-57/raw/traj_group0/traj0/policy_out.pkl"

def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]

def process_actions(path):  # gets actions
    fp = os.path.join(path, "policy_out.pkl")
    with open(fp, "rb") as f:
        act_list = pickle.load(f)
    if isinstance(act_list[0], dict):
        act_list = [x["actions"] for x in act_list]
    return act_list #arrays of 7 elements each

search_path = os.path.join(path, "raw", "traj_group*", "traj*")
print(search_path)
all_traj = glob.glob(search_path)
if all_traj == []:
    logging.info(f"no trajs found in {search_path}")
    exit()

random.shuffle(all_traj)

num_traj = len(all_traj)
for itraj, tp in tqdm.tqdm(enumerate(all_traj)):
    acts = process_actions(tp)
    state, next_state = process_state(tp)
    print(acts)