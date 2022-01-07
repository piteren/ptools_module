"""

 2021 (c) piteren

    QLearningNN

"""
import numpy as np

from ptools.R4C.qlearning.qlearning_interface import QLI
from ptools.R4C.renvy import QLearningEnvironment, EnvyState
from ptools.R4C.qlearning.qnn.qnn_graph import qnn_graph
from ptools.neuralmess.nemodel import NEModel


# QNeuralNetwork for QLearningEnvironment, builds (solves) given envy
class QNN(QLI):

    def __init__(
            self,
            envy: QLearningEnvironment,
            seed=   121,
            **kwargs):

        mdict = kwargs['mdict']
        if 'seed' in mdict: seed = mdict['seed']
        super().__init__(envy,seed)

        mdict['num_actions'] = self._envy.num_actions()
        mdict['num_states'] = self._envy.num_states()

        self.nn = NEModel(
            fwd_func=   qnn_graph,
            mdict=      mdict,
            save_TFD=   '_models',
            verb=       1)
        self.l_list = [] # stores losses while training

    # returns QV for given state
    def _get_QVs(self, state: EnvyState) -> np.ndarray:
        sid = self._envy.encode_state(state)
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['state_PH']: [sid]})
        return output[0] # reduce dim

    def _update(self, memory: QLI.ReplayMemory, batch_size=10, gamma=0.9):

        batch = memory.sample(batch_size)

        sidL = [self._envy.encode_state(e['next_state']) for e in batch]
        next_state_qvs = self.nn.session.run( # get QVs of next states (gold)
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['state_PH']: np.array(sidL)})

        # set Q-value to 0 for terminal states
        zeros = np.zeros(self._envy.num_actions())
        for i in range(len(batch)):
            if batch[i]['game_over']:
                next_state_qvs[i] = zeros

        _, loss = self.nn.session.run(
            fetches=[
                self.nn['optimizer'],
                self.nn['loss']],
            feed_dict={
                self.nn['state_PH']:        np.array([self._envy.encode_state(e['state']) for e in batch]),
                self.nn['reward_PH']:       np.array([e['reward'] for e in batch]),
                self.nn['enum_actions_PH']: np.array(list(enumerate([e['action'] for e in batch]))),
                self.nn['qv_target_PH']:    next_state_qvs})
        self.l_list.append(loss)