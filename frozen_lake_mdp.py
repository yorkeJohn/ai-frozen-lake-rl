'''
Author: John Yorke
CSCI 3482: Artificial Intelligence
Saint Mary's University
'''

from typing import List, Dict
import random

class FrozenLakeMDP:
    '''
    A class representing an MDP which implements methods for value and policy iteration
    '''
    def __init__(self, map: List[str], terminals: Dict[str, float], e: float, g: float, r: float, n: float) -> None:
        '''
        Params:
            map (List): The map of the grid world
            terminals (Dict): Dict defining which states are terminal and their rewards
            e (float): Error tolerance (epsilon)
            g (float): Discount (gamma)
            r (float): Non-terminal state (living) reward
            n (float): Noise, the total probability of ending up at an unexpected state
        '''
        self.map: List[str] = map
        self.rows: int = len(map)
        self.cols: int = len(map[0])
        self.S: List[tuple] = [(i, j) for i in range(self.rows) for j in range(self.cols)] # set of states
        self.terminals: Dict[tuple, float] = {(i, j): terminals[type] for i, j in self.S if (type := map[j][i]) in terminals}

        self.e: float = e
        self.g: float = g
        self.r: float = r
        self.n: float = n

        self.V_counter = 0
        self.pi_counter = 0

    def R(self, s: tuple) -> float:
        '''
        Reward function

        Params:
            s (tuple): The state
        Returns:
            float: The reward from transitioning to a state
        '''
        return self.terminals[s] if self.is_terminal(s) else self.r
    
    def T(self, s: tuple, a: tuple) -> List[tuple[tuple, float]]:
        '''
        Transition function

        Params:
            s (tuple): The state
            a (tuple): The action
        Returns:
            List: a list of transitions as tuples containing a state and its probability
        '''
        x, y = a
        sx, sy = s
        s_prime = (sx + x, sy + y), 1-self.n
        s_left = (sx + y, sy + x), self.n / 2
        s_right = (sx - y, sy - x), self.n / 2
        return [(new_s, p) if new_s in self.S else (s, p) for new_s, p in [s_prime, s_left, s_right]]
    
    def A(self, s: tuple) -> List[tuple]:
        '''
        Action function

        Params:
            s (tuple): The state
        Returns:
            List: a list of available actions from a given state
        '''
        A = [(0, 1), (1, 0), (0, -1), (-1, 0)] # up right down left
        return [] if self.is_terminal(s) else A

    def is_terminal(self, s: tuple) -> bool:
        '''
        Helper function to check if a state is terminal

        Params:
            s (tuple): The state
        Returns:
            bool: True if the state is terminal, otherwise False
        '''
        return s in self.terminals.keys()
    
    def Q(self, V: Dict[tuple, float], s: tuple, a: tuple) -> float:
        '''
        Computes a Q-value

        Params:
            V (Dict): The existing values
            s (tuple): The state
            a (tuple): The action
        Returns:
            float: a Q-value
        '''
        if not a:
            return 0
        return sum(p * V[s_prime] for s_prime, p in self.T(s, a))

    def value_iteration(self, V: Dict[tuple, float]=None) -> Dict[tuple, float]:
        '''
        Value iteration algorithm

        Params:
            V (Dict): The existing values (default None)
        Returns:
            Dict: Optimal values for each state
        '''
        self.V_counter += 1
        V = V or {s: 0 for s in self.S}
        V_next = {}
        d = 0
        for s in self.S:
            V_next[s] = self.R(s) + self.g * max([self.Q(V, s, a) for a in self.A(s)], default=0)
            d = max(d, abs(V_next[s] - V[s]))
        if d > self.e:
            V_next = self.value_iteration(V_next)
        return V_next
    
    def policy_evaluation(self, pi: Dict[tuple, tuple], V: Dict[tuple, float]=None) -> Dict[tuple, float]:
        '''
        Policy evaluation algorithm

        Params:
            pi (Dict): A policy
            V (Dict): The existing values (default None)
        Returns:
            Dict: values for each state
        '''
        self.pi_counter += 1
        V = V or {s: 0 for s in self.S}
        V_next = {}
        d = 0
        for s in self.S:
            V_next[s] = self.R(s) + self.g * self.Q(V, s, pi[s])
            d = max(d, abs(V_next[s] - V[s]))
        if d > self.e:
            V_next = self.policy_evaluation(pi, V_next)
        return V_next

    def policy_extraction(self, V: Dict[tuple, float]) -> Dict[tuple, tuple]:
        '''
        Policy extraction algorithm

        Params:
            V (Dict): Given values
        Returns:
            Dict: a policy
        '''
        return {s: max(self.A(s), key=lambda a: self.Q(V, s, a), default=None) for s in self.S}

    def policy_iteration(self, pi: Dict[tuple, tuple]=None) -> Dict[tuple, float]:
        '''
        Policy iteration algorithm

        Params:
            pi (Dict): The existing policy (default None)
        Returns:
            Dict: optimal values for each state
        '''
        self.pi_counter += 1
        pi = pi or {s: random.choice(self.A(s)) if self.A(s) else None for s in self.S}
        V = self.policy_evaluation(pi)
        new_pi = self.policy_extraction(V)
        if not all(pi[s] == new_pi[s] for s in self.S):
            V = self.policy_iteration(new_pi)
        return V