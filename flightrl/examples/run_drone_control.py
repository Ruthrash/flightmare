#!/usr/bin/env python3

import os
import math
import argparse
import tensorflow as tf
import numpy as np

#
from stable_baselines import logger

#
from rpg_baselines.common.policies import MlpPolicy
from rpg_baselines.common.util import ConfigurationSaver, TensorboardLauncher
from rpg_baselines.ppo.ppo2 import PPO2
from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
#
from flightgym import QuadrotorEnv_v1

#


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quad_env_cfg', type=str, default=os.path.abspath("../../configs/env.yaml"),
                        help='configuration file of the quad environment')
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--env', type=str, default="QuadrotorEnv_v1",
                        help="environment name")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('--load_dir', type=str,
                        help="Directory where to load weights")
    parser.add_argument('-m', '--mode', type=str, default='train',
                        help='set mode either train or test')
    parser.add_argument('-w', '--weight', type=str, default='./saved/2020-08-28-13-54-34_Iteration_393.zip',
                        help='trained weight path')
    return parser


def main():
    args = parser().parse_args()
    mode = args.mode

    env = wrapper.FlightEnvVec(QuadrotorEnv_v1())

    # set random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    #
    if mode == 'train':
        # save the configuration and other files
        rsg_root = os.path.dirname(os.path.abspath(__file__))
        log_dir = rsg_root + '/saved'
        saver = ConfigurationSaver(log_dir=log_dir)
        model = PPO2(
            tensorboard_log=saver.data_dir,
            policy=MlpPolicy,  # check activation function
            policy_kwargs=dict(
                net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
            env=env,
            lam=0.95,
            gamma=0.99,  # lower 0.9 ~ 0.99
            # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
            n_steps=250,
            ent_coef=0.00,
            learning_rate=3e-4,
            vf_coef=0.5,
            max_grad_norm=0.5,
            nminibatches=1,
            noptepochs=10,
            cliprange=0.2,
            verbose=1,
        )

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        logger.configure(folder=saver.data_dir)
        model.learn(
            total_timesteps=int(25000000),
            log_dir=saver.data_dir, logger=logger)
        model.save(saver.data_dir)

    # # Testing mode with a trained weight
    else:
        model = PPO2.load(args.weight)
        test_model(env, model, render=True)


if __name__ == "__main__":
    main()
