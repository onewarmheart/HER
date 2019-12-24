# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:31:57 2019

@author: Hill
"""

import gc
import psutil
import numpy as np
import logging 
def evaluating(args, env, agent, episode):
#    logging.basicConfig(filename=args.dir_name + '/her.log',level=logging.DEBUG,filemode='w')
    if(args.mode == 'test'):
        agent.load_models(args.load_path, episode)
    observation = env.reset()
    ep_r = 0
    for r in range(args.max_steps):
        if(args.render): env.render()
        state = np.float32(observation)
        action = agent.get_exploitation_action(state)

        new_observation, reward, done, info = env.step(action)
        
        ep_r = ep_r + reward

        observation = new_observation
        
        if done:
            break
        
    print("One episode test's Return: {}".format(ep_r))
    logging.info("One episode test's Return: {}".format(ep_r))