import numpy as np
from cogitation.trainer import Trainer
from cogitation.discover import Discover


class COGITATION(object):
    name = 'COGITATION'

    def __init__(self, args):
        self.model_path = args['model_path']
        self.model_id = args['model_id']
        args['trainer']['env_params'] = args['env_params']
        args['trainer']['cogitation_model'] = args['cogitation_model']
        args['discover']['env_params'] = args['env_params']

        # use discovered graph or not. use gt if not use discovered graph
        self.use_discover = True
        
        # only use causal when we use causal model
        if args['cogitation_model'] != 'causal':
            self.use_discover = False

        args['trainer']['use_discover'] = self.use_discover
        args['trainer']['use_gt'] = not self.use_discover

        # two modules
        self.trainer = Trainer(args['trainer'])
        self.discover = Discover(args['discover'])

        # decide the ratio between generation and discovery (generation is always longer)
        self.stage = 'generation'
        self.episode_counter = 0
        self.discovery_interval = args['discover']['discovery_interval']

    def stage_scheduler(self):
        if (self.episode_counter + 1) % self.discovery_interval == 0:
            self.stage = 'discovery'
        else:
            self.stage = 'generation'
        self.episode_counter += 1

    def select_action(self, env, state, deterministic):
        return self.trainer.select_action(env, state, deterministic)

    def store_transition(self, data):
        self.trainer.store_transition(data)
        self.discover.store_transition(data)

    def train(self):
        # discovery
        if self.stage == 'discovery' and self.use_discover:
            self.discover.update_causal_graph()
            self.trainer.set_causal_graph(self.discover.get_adj_matrix_graph())

        # generation
        self.trainer.train()

        # in the end, update the stage
        self.stage_scheduler()

    def save_model(self):
        self.trainer.save_model(self.model_path, self.model_id)
        self.discover.save_model(self.model_path, self.model_id)

    def load_model(self):
        self.trainer.load_model(self.model_path, self.model_id)
        self.discover.load_model(self.model_path, self.model_id)
