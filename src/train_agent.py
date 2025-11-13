from env import ColorChangingRL
from env import ColorChangingNoise
from agents import SAC
import matplotlib.pyplot as plt
from cogitation import COGITATION
import numpy as np
from utils import load_config
import copy
import argparse
import time
import os
np.set_printoptions(linewidth=np.inf)
import wandb
      
# parse arguments test
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='T')
parser.add_argument('--mode', type=str, required=True, choices=['IID', 'OOD-S'], help='IID means i.i.d. samples and OOD-S means spurious correlation')

parser.add_argument('--agent', type=str, default='COGITATION', choices=['COGITATION', 'SAC'])
parser.add_argument('--cogitation_model', type=str, default='mlp', choices=['causal', 'counterfact', 'mlp', 'gnn'], help='type of model used in COGITATION')

parser.add_argument('--env', type=str, default='chemistry', help='name of environment')
parser.add_argument('--graph', type=str, default='chain', choices=['collider', 'chain', 'full', 'jungle'], help='type of groundtruth graph in chemistry')

parser.add_argument('--noise_objects', type=int, default=2, help='number of objects that are noisy')

parser.add_argument('--use_state_abstraction', action='store_true', help='whether to use state abstraction')

args = parser.parse_args()

# args.exp_id = f"{args.mode}_{args.agent}_{args.cogitation_model}_m{args.env}_v{args.graph}"
# wandb.init(project='cogitation', name=args.exp_id, entity="mingatum")
args.exp_name = f"{args.mode}_{args.cogitation_model}_{args.graph}_{args.noise_objects}"
# environment parameters
if args.env == 'chemistry':
    num_steps = 10
    movement = 'Static' # Dynamic, Static
    if args.mode == 'IID':
        num_objects = 5
        num_colors = 5
    else:
        num_objects = 5
        num_colors = 5
    width = 5
    height = 5
    graph = args.graph + str(num_objects) # chain, full

    env = ColorChangingNoise(
            test_mode=args.mode, 
            render_type='shapes', 
            num_objects=num_objects, 
            noise_objects=args.noise_objects,
            use_state_abstraction = args.use_state_abstraction,
            num_colors=num_colors, 
            movement=movement, 
            max_steps=num_steps
    )

    env.set_graph(graph)

    config = load_config(config_path="config/chemistry_config.yaml")
    agent_config = config[args.agent]
    if args.use_state_abstraction:
        env_params = {
            'action_dim': env.action_space.n,
            'num_colors': env.num_colors,
            'num_objects': env.num_objects,
            'noise_objects': env.noise_objects,
            'width': env.width,
            'height': env.height,
            'state_dim': env.num_colors * env.num_objects * env.width * env.height * 2,
            'goal_dim': env.num_colors * env.num_objects * env.width * env.height,
            'adjacency_matrix': env.adjacency_matrix, # store the graph 
            'use_state_abstraction': env.use_state_abstraction
        }

    else:
        env_params = {
            'action_dim': env.action_space.n,
            'num_colors': env.num_colors,
            'num_objects': env.num_objects,
            'noise_objects': env.noise_objects,
            'width': env.width,
            'height': env.height,
            'state_dim': env.num_colors * (env.num_objects + env.noise_objects) * env.width * env.height * 2,
            'goal_dim': env.num_colors * (env.num_objects + env.noise_objects) * env.width * env.height,
            'adjacency_matrix': env.adjacency_matrix, # store the graph 
            'use_state_abstraction': env.use_state_abstraction
        }
    episode = 200
    test_episode = 100
else:
    raise ValueError('Wrong environment name')
env_params['env_name'] = args.env
agent_config['env_params'] = env_params
save_path = os.path.join('./log/baseline', args.exp_name)
# save_path = os.path.join('./log', args.exp_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

render = False
trails = 10
test_interval = 10
save_interval = 10000


if __name__ == '__main__':
    for t_i in range(trails):
        # create agent
        if args.agent == 'COGITATION':
            agent_config['cogitation_model'] = args.cogitation_model
            agent = COGITATION(agent_config)
        elif args.agent == 'SAC':
            agent = SAC(agent_config)

        save_gif_count = 0
        test_reward = []
        train_reward = []
        for e_i in range(episode):
            state = env.reset(stage='train')

            done = False
            one_train_reward = 0
            while not done:
                action = agent.select_action(env, state, False)
                next_state, reward, done, info = env.step(action)
                one_train_reward += reward

                if agent.name in ['SAC']: 
                    agent.store_transition([state, action, reward, next_state, done])
                    agent.train()
                elif agent.name in ['COGITATION']:
                    agent.store_transition([state, action, next_state])

                state = copy.deepcopy(next_state)

            if agent.name in ['COGITATION']: 
                agent.train()
            train_reward.append(one_train_reward)

            # save model
            if (e_i+1) % save_interval == 0:
                agent.model_id = e_i + 1
                agent.save_model()

            if (e_i+1) % test_interval == 0:
                test_reward_mean = []
                for t_j in range(test_episode):
                    state = env.reset(stage='test')
                    done = False
                    total_reward = 0
                    step_reward = []
                    while not done:
                        action = agent.select_action(env, state, True)
                        next_state, reward, done, info = env.step(action)

                        if render:
                            rendered_image = env.render()
                            if rendered_image.shape[0] == 6:
                                img1, img2 = np.split(rendered_image, 2, axis=0)                      
                                img1 = np.transpose(img1, (1, 2, 0))
                                img2 = np.transpose(img2, (1, 2, 0))
                                combined_image = np.concatenate((img1, img2), axis=1)
                                plt.imshow(combined_image)
                                plt.axis('off')  # hide the axis
                                plt.show()
                            time.sleep(1)
                        state = copy.deepcopy(next_state)
                        total_reward += reward
                        step_reward.append(reward)
                    test_reward_mean.append(total_reward)
                    
                test_reward_mean = np.mean(test_reward_mean, axis=0)
               
                print('[{}/{}] [{}/{}] Test Reward: {}'.format(t_i, trails, e_i, episode, test_reward_mean))
                test_reward.append(test_reward_mean)
                
                np.save(os.path.join(save_path, 'tower.test.reward.'+str(t_i)+'.npy'), test_reward)
