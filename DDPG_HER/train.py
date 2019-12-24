# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:26:13 2019

@author: Hill
"""


import numpy as np
from plot_tools import plot_reward

import gc
import psutil
import logging 

from evaluate import evaluating
def training(args, env, agent, ram, env_params):

    return_history = []
    for _ep in range(args.max_episodes):
        if(_ep % args.evaluate_interval == 0):
            evaluating(args, env, agent, 0)
            
        observation = env.reset()
        print('EPISODE :- ', _ep)
        ep_r = 0

        for r in range(args.max_steps):
#            env.render()
            state = np.float32(observation)
    #        state = observation
            action = agent.get_exploration_action(state)
            # if _ep%5 == 0:
            #     # validate every 5th episode
            #     action = agent.get_exploitation_action(state)
            # else:
            #     # get action based on observation, use exploration policy here
            #     action = agent.get_exploration_action(state)
    
            new_observation, reward, done, info = env.step(action)
            
            ep_r = ep_r + reward
            # # dont update if this is validation
            # if _ep%50 == 0 or _ep>450:
            #     continue
    
            if done:
                new_state = None
            else:
    #            new_state = new_observation
                new_state = np.float32(new_observation)
                # push this exp in ram
                ram.add(state, action, reward, new_state)
    
            observation = new_observation
    
            # perform optimization
            agent.optimize()
            if done:
                break
        
        print("Episode: {} | Return: {}".format(_ep, ep_r))
        logging.info("Episode: {} | Return: {}".format(_ep, ep_r))
        return_history.append(ep_r)
        # check memory consumption and clear memory
        gc.collect()
        # process = psutil.Process(os.getpid())
        # print(process.memory_info().rss)
    
        if _ep == 0 or _ep % 99 == 0:
            agent.save_models(args.dir_name, _ep)
    plot_reward(return_history, args.dir_name)
    
    print('Completed episodes')