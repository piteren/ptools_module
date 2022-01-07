"""

 2021 (c) piteren

    reinforcement environment interfaces

"""
from abc import abstractmethod, ABC
import numpy as np
from typing import List


# generic RL state
class EnvyState(ABC):
    pass


# base interface of reinforcement environment
class REnvy(ABC):

    # returns current state
    @abstractmethod
    def get_state(self) -> EnvyState: pass

    # plays action (goes to new state, returns reward)
    @abstractmethod
    def run(self, action: int) -> float: pass

    # checks if envy(episode) is over
    @abstractmethod
    def is_over(self) -> bool: pass

    # resets envy
    @abstractmethod
    def reset(self): pass

    # override to implement envy rendering (for debug, preview etc.)
    def render(self): pass


# interface of reinforcement environment with finite number of states
class FiniteActionsREnvy(REnvy):

    # returns number of envy actions
    @abstractmethod
    def num_actions(self) -> int: pass


# interface of reinforcement environment with finite number of states
class FiniteStatesREnvy(REnvy):

    # returns number of envy states
    @abstractmethod
    def num_states(self) -> int: pass


# interface of reinforcement environment that supports self encoding for RL solver (qtable, qnn, pgnn)
class SolverSupportingREnvy(REnvy):

    # encodes EnvyState to type accepted by RL solver
    @abstractmethod
    def encode_state(self, state: EnvyState): pass


# reinforcement environment for QLearning
class QLearningEnvironment(FiniteActionsREnvy,FiniteStatesREnvy,SolverSupportingREnvy):

    # encodes EnvyState to type accepted by QLearning solver (hashes state to int)
    @abstractmethod
    def encode_state(self, state: EnvyState) -> int: pass

    # returns all possible states (for testing purposes)
    @abstractmethod
    def get_all_states(self) -> List[EnvyState]: pass

    # returns evaluation score of state and action
    @abstractmethod
    def evaluate(self, state: EnvyState, action: int) -> float: pass

# reinforcement environment for Policy Gradients
class PolicyGradientsEnvironment(FiniteActionsREnvy,SolverSupportingREnvy):

    # encodes EnvyState to type accepted by Policy Gradients solver
    @abstractmethod
    def encode_state(self, state: EnvyState) -> np.array: pass

    def get_state_width(self) -> int:
        state = self.get_state()
        encoded_state = self.encode_state(state)
        return encoded_state.shape[0]