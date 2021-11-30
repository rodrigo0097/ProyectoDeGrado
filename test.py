import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv

# env = gym.make('CartPole-v1')
# env = DummyVecEnv([lambda:env])
#
# # model = PPO2(MlpPolicy, env, verbose=1)
# # model.learn(total_timesteps=10000)
# # model.save("cartpole-ppo2")
#
# model = PPO2.load("cartpole-ppo2")
# model.set_env(env)
# obs = env.reset()
#
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

import retro

def main():
    env = retro.make(game='IceClimber-Nes')
    env = DummyVecEnv([lambda:env])
    model = PPO2(CnnPolicy, env, verbose=1 )
    model.learn(total_timesteps=15)
    model.save("iceClimbers")

    # model = PPO2.load("iceClimbers")
    # model.set_env(env)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    # while True:
    #     obs, rew, done, info = env.step(env.action_space.sample())
    #     env.render()
    #     if done:
    #         obs = env.reset()
    # env.close()


if __name__ == "__main__":
    main()