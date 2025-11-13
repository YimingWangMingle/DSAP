import numpy as np


class ReplayBuffer():
    def __init__(self, memory_capacity, buffer_dim):
        self.buffer_dim = buffer_dim
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, self.buffer_dim))
        self.memory_counter = 0
        self.memory_len = 0

    def push(self, data):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = data
        self.memory_counter += 1
        self.memory_len = min(self.memory_len+1, self.memory_capacity)

    def sample(self, batch_size):
        sample_index = np.random.randint(0, self.memory_len, size=batch_size)
        batch_memory = self.memory[sample_index, :]
        return batch_memory
