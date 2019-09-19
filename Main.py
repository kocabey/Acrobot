from Acrobot import Acrobot
from PPO import learn

learn(
    Acrobot(),
    horizon=2048,
    clip=0.1,
    epochs=10,
    learning_rate=1e-4,
    max_std=2.5,
    min_std=0.5,
    std_iterations=2500,
    batch_size=128,
    num_hidden=3,
    hidden_size=256,
    save_interval=50,
    experiment_name="acrobot"
)
