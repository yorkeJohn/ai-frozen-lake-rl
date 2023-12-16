'''
Author: John Yorke
CSCI 3482: Artificial Intelligence
Saint Mary's University
'''

from frozen_lake_mdp import FrozenLakeMDP
from tabulate import tabulate
from typing import Dict, List
import gymnasium as gym

def show_V(V: Dict[tuple, float], rows: int) -> None:
    '''
    Helper method to display values
    '''
    data = [list(V.values())[i::rows] for i in range(rows)]
    print(tabulate(data, floatfmt='.5f', tablefmt='simple_grid'))

def show_pi(pi: Dict[tuple, tuple], rows: int) -> None:
    '''
    Helper method to display a policy
    '''
    mapping = {(0, 1): '\u2193', (1, 0): '\u2192', (0, -1): '\u2191', (-1, 0): '\u2190', None: '\u25A0'}
    data = [map(lambda a: mapping[a], d) for d in [list(pi.values())[i::rows] for i in range(rows)]]
    print(tabulate(data, tablefmt='simple_grid'))

def compile_pi(pi: Dict[tuple, tuple], rows: int, cols: int) -> List[int]:
    '''
    Converts a policy to be usable by Gym
    '''
    mapping = {(0, 1): 1, (1, 0): 2, (0, -1): 3, (-1, 0): 0, None: -1}
    data = [list(pi.values())[i::rows] for i in range(rows)]
    return list(map(lambda a: mapping[a], [data[j][i] for j in range(rows) for i in range(cols)]))

if __name__ == '__main__':
    # S:Start, F:Frozen, H:Hole, G:Goal
    world: List[str] = ["SFFF", "FHFH", "FFFF", "HFFG"]
    # params
    slippery: bool = False
    n: float = 2/3 if slippery else 0 # in a non-deterministic world we slip 2/3 of the time

    mdp = FrozenLakeMDP(e=10E-3, g=0.8, r=0, n=n, map=world, terminals={'G': 1, 'H': 0})
    env = gym.make('FrozenLake-v1', render_mode="human", desc=world, map_name="4x4", is_slippery=slippery)

    # use value iteration
    V = mdp.value_iteration()
    print(f'V* values using value iteration (required {mdp.V_counter} iterations):')
    show_V(V, mdp.rows)

    pi = mdp.policy_extraction(V)
    print('Policy \u03C0* extracted from V* using value iteration:')
    show_pi(pi, mdp.rows)
    
    # use policy iteration
    V2 = mdp.policy_iteration()
    print(f'V* values using policy iteration (required {mdp.pi_counter} iterations):')
    show_V(V2, mdp.rows)

    pi2 = mdp.policy_extraction(V2)
    print('Policy \u03C0* extracted from V* using policy iteration:')
    show_pi(pi2, mdp.rows)

    # test the policy
    env.reset()
    env.render()
    policy = compile_pi(pi, mdp.rows, mdp.cols)

    s = 0
    goal = 15
    while s != goal:
        a = policy[s]
        s, r, t, f, p = env.step(a)
        if t and s != goal:
            env.reset()
            s = 0
    print('Done')
