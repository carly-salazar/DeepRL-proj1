# python3 - <<’PY’
import gymnasium as gym
import ale_py
# Classic control
env = gym.make("CartPole-v1")
env.reset()
env.close()
print("CartPole OK")
# Box2D
env = gym.make("LunarLander-v3")fi
env.reset()
env.close()
print("LunarLander OK")
# Atari / ALE
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5")
env.reset()
env.close()
print("Pong OK")
# PY