from abc import abstractmethod, ABC
import numpy as np


from ptools.lipytools.little_methods import stamp
from ptools.R4C.actor import Actor
from ptools.R4C.policy_gradients.a2c.a2c_graph import a2c_graph
from ptools.neuralmess.nemodel import NEModel


class A2CModel(Actor, ABC):

    def __init__(
            self,
            observation,
            num_actions: int,
            mdict :dict,
            graph=      a2c_graph,
            devices=    -1,
            verb=       1):

        self.verb = verb

        self.num_actions = num_actions
        observation_width = self.observation_vec(observation).shape[-1]

        if self.verb>0: print(f'\n*** ACSharedModel inits, observation_width: {observation_width}')

        mdict['observation_width'] = observation_width
        mdict['num_actions'] = self.num_actions
        if 'name' not in mdict: mdict['name'] = f'a2c_{stamp()}'

        self.nn = NEModel(
            fwd_func=       graph,
            devices=        devices,
            save_topdir=    '_models',
            verb=           self.verb-1,
            **mdict)

        self.__upd_step = 0

    # returns vector of feats from given observation (encodes observation into input vector)
    @abstractmethod
    def observation_vec(self, observation) -> np.ndarray: pass

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def observation_vec_batch(self, observations) -> np.ndarray:
        return np.array([self.observation_vec(v) for v in observations])

    # USED for step by step actions of Actor
    def get_policy_probs(self, observation) -> np.ndarray:
        ov = self.observation_vec(observation)
        probs = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: [ov]},
            fetches=    self.nn['action_prob'])
        return probs[0]

    def get_policy_probs_batch(self, observations) -> np.ndarray:
        ovs = self.observation_vec_batch(observations)
        probs = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: ovs},
            fetches=    self.nn['action_prob'])
        return probs

    def get_policy_action(self, observation, sampled=False) -> int:
        probs = self.get_policy_probs(observation)
        if sampled: action = np.random.choice(self.num_actions, p=probs)
        else:       action = int(np.argmax(probs))
        return action

    def get_qvs(self, observation) -> np.ndarray:
        ov = self.observation_vec(observation)
        qvs = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: [ov]},
            fetches=    self.nn['qvs'])
        return qvs

    def get_qvs_batch(self, observations) -> np.ndarray:
        ovs = self.observation_vec_batch(observations)
        qvss = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: ovs},
            fetches=    self.nn['qvs'])
        return qvss

    def get_values_batch(self, observations) -> np.ndarray:
        ovs = self.observation_vec_batch(observations)
        values = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: ovs},
            fetches=    self.nn['value'])
        return values

    def update_batch(
            self,
            observations,
            actions,
            dreturns) -> float:

        ovs = self.observation_vec_batch(observations)
        _, loss, loss_actor, loss_critic, gn, gn_avt, amax_prob, amin_prob, ace = self.nn.session.run(
            fetches=    [
                self.nn['optimizer'],
                self.nn['loss'],
                self.nn['loss_actor'],
                self.nn['loss_critic'],
                self.nn['gg_norm'],
                self.nn['gg_avt_norm'],
                self.nn['amax_prob'],
                self.nn['amin_prob'],
                self.nn['actor_ce_mean']],
            feed_dict=  {
                self.nn['observation_PH']:  ovs,
                self.nn['action_PH']:       actions,
                self.nn['return_PH']:       dreturns})

        self.__upd_step += 1

        if self.verb>0:
            self.nn.log_TB(loss,        'upd/loss',             step=self.__upd_step)
            self.nn.log_TB(loss_actor,  'upd/loss_actor',       step=self.__upd_step)
            self.nn.log_TB(loss_critic, 'upd/loss_critic',      step=self.__upd_step)
            self.nn.log_TB(gn,          'upd/gn',               step=self.__upd_step)
            self.nn.log_TB(gn_avt,      'upd/gn_avt',           step=self.__upd_step)
            self.nn.log_TB(amax_prob,   'upd/amax_prob',        step=self.__upd_step)
            self.nn.log_TB(amin_prob,   'upd/amin_prob',        step=self.__upd_step)
            self.nn.log_TB(ace,         'upd/actor_ce_mean',    step=self.__upd_step)

        return loss_actor