"""

 2021 (c) piteren

    PolicyGradientsNN

    TODO:
     - add mov-avg for TB
     - implement parallel training, in batches (many envys)

"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.utils import shuffle
from typing import List

from ptools.R4C.renvy import PolicyGradientsEnvironment
from ptools.R4C.policy_gradients.pgnn_graph import pgnn_graph
from ptools.neuralmess.nemodel import NEModel


def discounted_acc_rewards(rewards: List[float], gamma: float) -> List[float]:

    dar = np.zeros_like(rewards)
    s = 0.0
    for i in reversed(range(len(rewards))):
        s = s * gamma + rewards[i]
        dar[i] = s

    """
    z-score is (x-mean(x))/stddev(x)
    this is (?) helpful for training, as rewards can vary considerably between episodes,
    which will have a bad impact over the loss minimization
    """
    return list(zscore(dar))


class PGNN:

    def __init__(self,
            envy: PolicyGradientsEnvironment,
            mdict :dict):

        self._envy = envy

        mdict['num_actions'] = self._envy.num_actions()
        mdict['state_size'] = self._envy.get_state_width()


        self._nn = NEModel(
            fwd_func=   pgnn_graph,
            mdict=      mdict,
            save_TFD=   '_models',
            verb=       1)

    def run_episode(
            self,
            exploration=    True,
            render=         True):

        self._envy.reset()
        steps = 0
        states = []
        rewards = []
        actions = []
        while not self._envy.is_over():
            steps += 1
            current_state = self._envy.get_state()
            current_state_enc = self._envy.encode_state(current_state)
            current_state_enc_batch = np.expand_dims(current_state_enc, axis=0)
            probs_batch = self._nn.session.run(
                fetches=    self._nn['action_prob'],
                feed_dict=  {self._nn['states_PH']: current_state_enc_batch})
            if exploration: action = np.random.choice(self._envy.num_actions(), p=probs_batch.flatten())
            else:           action = int(np.argmax(probs_batch.flatten()))
            r = self._envy.run(action)

            # save to memory:
            states.append(current_state)
            rewards.append(r)
            actions.append(action)

            if render: self._envy.render()

        return states, rewards, actions, steps

    # train with exploration
    def train(
            self,
            num_episodes=   1500,
            exploration=    True,
            gamma=          0.99,
            upd_batch_size= 32,
            do_TB=          True,
            debug=          100,
            render=         True):

        print(f'\nStarting train for {num_episodes} episodes...')
        data = pd.DataFrame(columns=['steps','cost'])

        upd_step = 0
        upd_states, upd_da_rewards, upd_actions = [], [], []
        for ep in range(num_episodes):

            states, rewards, actions, steps = self.run_episode(
                exploration=    exploration,
                render=         render and ep%debug==0)

            da_rewards = discounted_acc_rewards(rewards, gamma)

            states, da_rewards, actions = shuffle(states, da_rewards, actions) # shuffle in a consistent way

            upd_states +=       states
            upd_da_rewards +=   da_rewards
            upd_actions +=      actions

            # update in batches
            n_upd = 0
            if len(upd_states) > upd_batch_size:
                six = 0
                while six+upd_batch_size <= len(upd_states):
                    loss, gn, agn, _ = self._nn.session.run(
                        fetches=    [
                            self._nn['loss'],
                            self._nn['gg_norm'],
                            self._nn['avt_gg_norm'],
                            self._nn['optimizer']],
                        feed_dict=  {
                            self._nn['states_PH']:   upd_states[six:six+upd_batch_size],
                            self._nn['acc_rew_PH']:  upd_da_rewards[six:six+upd_batch_size],
                            self._nn['actions_PH']:  upd_actions[six:six+upd_batch_size]})
                    six += upd_batch_size
                    if do_TB:
                        self._nn.log_TB(loss, 'tr/loss',        step=upd_step)
                        self._nn.log_TB(gn,   'tr/gg_norm',     step=upd_step)
                        self._nn.log_TB(agn,  'tr/avt_gg_norm', step=upd_step)
                    upd_step += 1
                    n_upd += 1
                upd_states, upd_da_rewards, upd_actions = [], [], [] # clear after updating (for new policy)

            if do_TB:
                self._nn.log_TB(steps, 'ep/steps', step=ep)
                self._nn.log_TB(n_upd, 'ep/n_upd', step=ep)
            data = data.append({'steps':steps}, ignore_index=True)

            if ep%debug==0: print(f'episode {ep} has ended after {steps} steps')

    # test with exploitation
    def test(
            self,
            exploration=    False,
            num_episodes=   10,
            debug=          1,
            render=         True):

        print(f'\nStarting tests for {num_episodes} episodes...')

        data = pd.DataFrame(columns=['steps'])
        for ep in range(num_episodes):

            _, _, _, steps = self.run_episode(
                exploration=    exploration,
                render=         render and ep%debug == 0)

            data = data.append({'steps': steps}, ignore_index=True)

            if ep%debug == 0: print(f'episode {ep} has ended after {steps} steps')

