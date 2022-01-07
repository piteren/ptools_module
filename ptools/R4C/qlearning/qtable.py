"""

 2021 (c) piteren

    Q-Table

"""
import numpy as np

from ptools.R4C.qlearning.qlearning_interface import QLI
from ptools.R4C.renvy import QLearningEnvironment, EnvyState


# QTable for REnvy, builds (solves) given envy(REnvy)
class QTable(QLI):

    def __init__(
            self,
            envy: QLearningEnvironment,
            seed=   123):

        super().__init__(envy, seed)
        self.__table = {} # {encoded_state(int): np.array(QValues)}

    def __init_state(self, sid: int):
        self.__table[sid] = np.random.random(self._envy.num_actions()) # init with random
        # self.__tbl[sid] = np.zeros(self.num_actions, dtype=np.float) # init with 0

    # returns QV for given state
    def _get_QVs(self, state: EnvyState) -> np.ndarray:
        sid = self._envy.encode_state(state)
        if sid not in self.__table: self.__init_state(sid)
        return self.__table[sid]

    # updates QV for given action and state
    def __upd_QV(self, state: EnvyState, action: int, qv: float):
        sid = self._envy.encode_state(state)
        if sid not in self.__table: self.__init_state(sid)
        self.__table[sid][action] = qv

    def _update(self, memory: QLI.ReplayMemory, batch_size=10, gamma=0.9):

        batch = memory.sample(batch_size)
        for e in batch:

            next_state_qvs = self._get_QVs(e['next_state'])
            next_state_max_qv = max(next_state_qvs)         # get max QV of next state

            new_qv = e['reward'] + gamma * next_state_max_qv
            self.__upd_QV(e['state'], e['action'], new_qv)  # update QV

    def __str__(self):
        s = 'QTable:\n'
        keys = sorted(list(self.__table.keys()))
        if keys:
            for st in keys:
                s += f'{st} : {self.__table[st]}\n'
        else: s += 'empty\n'
        return s