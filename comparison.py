from stable_baselines3 import DQN, PPO, SAC

models = {
    "DQN": DQN("MlpPolicy", env),
    "DDQN": DQN("MlpPolicy", env, double_q=True),
    "PPO": PPO("MlpPolicy", env),
    "SAC": SAC("MlpPolicy", env)
}

results = {}

for name, model in models.items():
    model.learn(80_000)
    rewards = evaluate_policy(model, env, n_eval_episodes=10)
    results[name] = rewards


import pandas as pd

comparison = pd.DataFrame({
    "Model": results.keys(),
    "Avg Reward": [r[0] for r in results.values()],
    "Std Reward": [r[1] for r in results.values()]
})

print(comparison)
