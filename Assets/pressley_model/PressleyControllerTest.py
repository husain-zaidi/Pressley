from mlagents_envs.environment import UnityEnvironment

print("Press play in Unity")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])

env.reset()
behavior_names = env.behavior_specs.keys()
print(behavior_names)
behavior_name = list(env.behavior_specs)[0]
print(behavior_name)
spec = env.behavior_specs[behavior_name]
print("Number of observations : ", len(spec.observation_specs))

decision_steps, terminal_steps = env.get_steps(behavior_name)
print(decision_steps)
print(terminal_steps)
env.set_actions(behavior_name, spec.action_spec.empty_action(len(decision_steps)))
env.step()
env.reset()


for episode in range(3):
  env.reset()
  decision_steps, terminal_steps = env.get_steps(behavior_name)
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  while not done:
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[0]

    # Generate an action for all agents
    action = spec.action_spec.random_action(len(decision_steps))

    # Set the actions
    env.set_actions(behavior_name, action)

    # Move the simulation forward
    env.step()

    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if tracked_agent in decision_steps: # The agent requested a decision
      episode_rewards += decision_steps[tracked_agent].reward
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Total rewards for episode {episode} is {episode_rewards}")

env.close()
print("Closed environment")
