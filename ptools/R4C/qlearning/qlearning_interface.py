"""

 2021 (c) piteren

    QLearning Interface - common for QTable and QNN

"""

from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import random
from typing import List

from ptools.R4C.renvy import QLearningEnvironment, EnvyState


class QLI(ABC):

    class ReplayMemory:

        def __init__(self, size):
            self.memory = deque(maxlen=size)
            self.counter = 0

        def append(self, element):
            self.memory.append(element)
            self.counter += 1

        def sample(self, n):
            return random.sample(self.memory, n)

    def __init__(
            self,
            envy: QLearningEnvironment,
            seed=               123,
            **kwargs):

        random.seed(seed)
        self._envy = envy

    # returns QV for given state
    @abstractmethod
    def _get_QVs(self, state: EnvyState) -> np.ndarray: pass

    # update for based on memory
    @ abstractmethod
    def _update(self, memory: ReplayMemory, batch_size=10, gamma=0.9): pass

    # Q-learning training procedure
    def train(
            self,
            num_of_games=   2000,
            epsilon=        0.5,
            gamma=          0.9,
            batch_size=     10,
            memory_size=    20) -> List[float]:

        memory = QLI.ReplayMemory(memory_size)

        r_list = []  # store the total reward of each game so we can plot it later
        counter = 0  # update trigger
        for g in range(num_of_games):
            total_reward = 0
            self._envy.reset()
            while not self._envy.is_over():
                state = self._envy.get_state()                             # save initial state copy

                if random.random() < epsilon:
                    action = random.randrange(self._envy.num_actions())    # select random action
                else:
                    qvs = self._get_QVs(state)                             # get all actions QVs
                    action = np.argmax(qvs)                                # select action with max QV

                reward = self._envy.run(action)                            # get it reward
                total_reward += reward

                memory.append({
                    'state':        state,
                    'action':       action,
                    'reward':       reward,
                    'next_state':   self._envy.get_state(),
                    'game_over':    self._envy.is_over()})

                counter += 1  # update counter to trigger training
                if counter % batch_size == 0: self._update(                # got batch >> update
                    memory=     memory,
                    batch_size= batch_size,
                    gamma=      gamma)

            r_list.append(total_reward)

        return r_list

    def test(self):
        all_states = self._envy.get_all_states()
        for st in all_states:
            qv = self._get_QVs(st)
            action = int(np.argmax(qv))
            pred = str([round(v, 3) for v in qv])
            sid = self._envy.encode_state(st)
            print(f'state: {sid}  QVs: {pred:30s}  action: {action}  (eval:{self._envy.evaluate(st, action)})')

