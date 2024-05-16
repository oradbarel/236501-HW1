import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple, Any
from typing_extensions import Self
import heapdict
from collections import deque


class Node():
    def __init__(self, state: Tuple[int, bool, bool], parent: Self = None, action: int = -1,
                 cost: float = 0.0, g_value: float = 0.0, h_value: int = 0) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.g_value = g_value
        self.h_value = h_value

    @property
    def state(self) -> Tuple[int, bool, bool]:
        return self._state

    @state.setter
    def state(self, state: Tuple[int, bool, bool]):
        if not isinstance(state, tuple) or not list(map(type, state)) == [int, bool, bool]:
            raise TypeError
        self._state = state

    @property
    def position(self) -> int:
        return self._state[0]

    @property
    def first_collected(self) -> bool:
        return self._state[1]

    @property
    def second_collected(self) -> bool:
        return self._state[2]

    @property
    def parent(self) -> Self:
        return self._parent

    @parent.setter
    def parent(self, parent: Self):
        if parent and not isinstance(parent, Node):
            raise TypeError
        self._parent = parent

    @property
    def action(self) -> int:
        return self._action

    @action.setter
    def action(self, action: int):
        if not isinstance(action, int):
            raise TypeError
        self._action = action

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, cost: float):
        if not isinstance(cost, (int, float)):
            raise TypeError
        self._cost = float(cost)

    @property
    def g_value(self) -> float:
        return self._g_value

    @g_value.setter
    def g_value(self, g_value: float):
        if not isinstance(g_value, (int, float)):
            raise TypeError
        self._g_value = float(g_value)

    @property
    def h_value(self) -> int:
        return self._h_value

    @h_value.setter
    def h_value(self, h_value: int):
        if not isinstance(h_value, int):
            raise TypeError
        self._h_value = h_value

    def is_source(self) -> bool:
        return not self.parent


class NodeHeapdict(heapdict.heapdict):
    @property
    def NODE_LOC(self) -> int:
        return 0

    @property
    def PRIORITY_LOC(self) -> int:
        return 1

    @property
    def F_VAL_LOC(self) -> int:
        return 0

    @property
    def POS_LOC(self) -> int:
        return 1

    def pop_node(self):
        return self.popitem()[self.NODE_LOC]

    def pop_priority(self):
        return self.popitem()[self.PRIORITY_LOC]

    def pop_f_val(self):
        return self.popitem()[self.PRIORITY_LOC][self.F_VAL_LOC]

    def pop_position(self):
        return self.popitem()[self.PRIORITY_LOC][self.POS_LOC]

    def peek_node(self):
        return self.peekitem()[self.NODE_LOC]

    def peek_priority(self):
        return self.peekitem()[self.PRIORITY_LOC]

    def peek_f_val(self):
        return self.peekitem()[self.PRIORITY_LOC][self.F_VAL_LOC]

    def peek_position(self):
        return self.peekitem()[self.PRIORITY_LOC][self.POS_LOC]


class Agent:
    def __init__(self) -> None:
        self.env: DragonBallEnv = None
        self.expanded_states: int = None

    def is_hole(self, node: Node) -> bool:
        if not isinstance(self.env, DragonBallEnv):
            raise AssertionError("env was not initialized yet.")
        return None in self.env.succ(node.state)[0]

    def is_final_state(self, node: Node) -> bool:
        if not isinstance(self.env, DragonBallEnv):
            raise AssertionError("env was not initialized yet.")
        return self.env.is_final_state(node.state)

    def terminated_and_not_final_state(self, node: Node) -> bool:
        """
        Ruturns true whether the agent reaches a final state without the 2 balls or falls into a hole
        """
        if not isinstance(self.env, DragonBallEnv):
            raise AssertionError("env was not initialized yet.")
        return self.is_hole(node) or (
            not self.is_final_state(node) and self.env.is_final_state((node.position, True, True)))

    @staticmethod
    def get_empty_solotion() -> Tuple[List[int], int, int]:
        return [], 0, 0

    def get_solution(self, node: Node) -> Tuple[List[int], float, int]:
        actions, cost = [], 0.0
        curr_node = node
        while curr_node.parent:
            actions.append(curr_node.action)
            cost += curr_node.cost
            curr_node = curr_node.parent
        return actions[::-1], cost, self.expanded_states

    def init_search(self, env: DragonBallEnv):
        self.env = env
        self.env.reset()
        self.expanded_states = 0

    @staticmethod
    def manhatan_dist(x: Tuple[int, int], y: Tuple[int, int]) -> int:
        if not all(isinstance(p, tuple) for p in [x, y]):
            raise TypeError
        return sum(abs(a - b) for a, b in zip(x, y))

    def manhatan_heuristic(self, state: Tuple[int, bool, bool]) -> int:
        h_goals = [g for g in self.env.get_goal_states() if g[1] and g[2]]
        h_goals += [self.env.d1] if not state[1] else []
        h_goals += [self.env.d2] if not state[2] else []
        return min(
            [self.manhatan_dist(self.env.to_row_col(state), self.env.to_row_col(g))
             for g in h_goals]
        )


class BFSAgent(Agent):
    def search(self, env: DragonBallEnv) -> Tuple[List[int], float | int, int]:
        super().init_search(env)
        # Check for start node:
        start_node = Node(self.env.get_initial_state())
        if self.is_final_state(start_node):
            return Agent.get_empty_solotion()
        # Create data structers:
        open_queue: deque[Node] = deque([start_node])
        close_set: set[Tuple[int, bool, bool]] = set()
        # Iterate while open is not empty:
        while open_queue:
            # Pop from `open` and insert to `close`:
            curr_node = open_queue.popleft()
            self.expanded_states += 1
            close_set.add(curr_node.state)
            # Skip holes and goals without 2 dragon balls:
            if self.terminated_and_not_final_state(curr_node):
                continue
            # Expand node and loop for the successors:
            for action in self.env.succ(curr_node.state):
                # Reset and set_state so we won't be affected by other successors:
                self.env.reset()
                self.env.set_state(curr_node.state)
                # Perform the step and create the new node:
                new_state, step_cost, _ = self.env.step(action)
                new_node = Node(state=new_state, parent=curr_node,
                                action=action, cost=step_cost)
                # In case new_node not in `open` and not in `close`:
                if new_node.state not in [n.state for n in open_queue] and new_node.state not in close_set:
                    if self.is_final_state(new_node):
                        return self.get_solution(new_node)
                    open_queue.append(new_node)
        return Agent.get_empty_solotion()


class WeightedAStarEpsilonAgent(Agent):
    def __init__(self):
        super().__init__()
        self.h_weight = 0.5
        self.epsilon = 0.0
        self.open_queue = NodeHeapdict()
        self.close_set: set[Node] = set()

    def init_search(self, env: DragonBallEnv):
        super().init_search(env)
        self.open_queue.clear()
        self.close_set.clear()

    @property
    def h_weight(self) -> float:
        return self._h_weight

    @h_weight.setter
    def h_weight(self, h_weight: float):
        if not isinstance(h_weight, (int, float)):
            raise TypeError
        self._h_weight = float(h_weight)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: float):
        if not isinstance(epsilon, (int, float)):
            raise TypeError
        self._epsilon = float(epsilon)

    @property
    def open_queue(self) -> NodeHeapdict:
        return self._open_queue

    @open_queue.setter
    def open_queue(self, open_queue: NodeHeapdict):
        if not isinstance(open_queue, NodeHeapdict):
            raise TypeError
        self._open_queue = open_queue

    @property
    def close_set(self) -> set:
        return self._close_set

    @close_set.setter
    def close_set(self, close_set: set):
        if not isinstance(close_set, set):
            raise TypeError
        self._close_set = close_set

    def priority(self, node: Node) -> tuple[float, int]:
        if not isinstance(self.env, DragonBallEnv) or not isinstance(self.h_weight, float):
            raise AssertionError("env was not initialized yet.")
        if not isinstance(node, Node):
            raise TypeError
        return (np.dot([node.g_value, node.h_value], [1-self.h_weight, self.h_weight]), node.position)

    def pop_from_open(self) -> Node:
        if self.epsilon:
            peek_f_val = self.open_queue.peek_f_val()
            focal_queue = NodeHeapdict()
            for curr_n, curr_p in self.open_queue.items():
                if curr_p[focal_queue.F_VAL_LOC] <= (1+self.epsilon) * peek_f_val:
                    focal_queue[curr_n] = curr_p
            min_node = focal_queue.pop_node()
            del self.open_queue[min_node]
            return min_node
        else:
            return self.open_queue.pop_node()

    def append_to_open(self, node: Node):
        self.open_queue[node] = self.priority(node)

    def super_search(self, env: DragonBallEnv, h_weight: float = 0.5, epsilon: float = 0) -> Tuple[List[int], float | int, int]:
        self.init_search(env)
        self.h_weight = h_weight
        self.epsilon = epsilon
        # Append initial state:
        start_state = self.env.get_initial_state()
        start_node = Node(state=start_state, g_value=0,
                          h_value=self.manhatan_heuristic(start_state))
        self.append_to_open(start_node)
        # Iterate while open is not empty:
        while self.open_queue:
            # Pop from `open` and insert to `close`:
            curr_node = self.pop_from_open()
            self.close_set.add(curr_node)
            if self.is_final_state(curr_node):
                return self.get_solution(curr_node)
            self.expanded_states += 1
            # Do not expand `hole` nodes:
            if self.terminated_and_not_final_state(curr_node):
                continue
            for action in self.env.succ(curr_node.state):
                # Reset and set_state so we won't be affected by other successors:
                self.env.reset()
                self.env.set_state(curr_node.state)
                # Perform the step and create the new node:
                new_state, step_cost, _ = self.env.step(action)
                new_node = Node(state=new_state, parent=curr_node, action=action, cost=step_cost,
                                g_value=curr_node.g_value+step_cost, h_value=self.manhatan_heuristic(new_state))
                # Switch-case over "new_node in open" and "new_node in close":
                new_node_in_open = [
                    n for n in self.open_queue if n.state == new_node.state]
                new_node_in_close = [
                    n for n in self.close_set if n.state == new_node.state]
                """ Probably need to remove
                # For holes - just append to close_set:
                if self.terminated_and_not_final_state(new_node):
                    self.close_set.add(new_node)
                    continue
                """
                # In case new_node not in `open` and not in `close`:
                if not new_node_in_open and not new_node_in_close:
                    self.append_to_open(new_node)
                elif new_node_in_open:
                    existed_node = new_node_in_open[self.open_queue.NODE_LOC]
                    if self.priority(new_node) < self.priority(existed_node):
                        del self.open_queue[existed_node]
                        self.append_to_open(new_node)
                else:  # new_node_in_close
                    existed_node = new_node_in_close[self.open_queue.NODE_LOC]
                    if self.priority(new_node) < self.priority(existed_node):
                        self.close_set.remove(existed_node)
                        self.append_to_open(new_node)
        return Agent.get_empty_solotion()


class WeightedAStarAgent(WeightedAStarEpsilonAgent):
    def search(self, env: DragonBallEnv, h_weight: float) -> Tuple[List[int], float, int]:
        return super().super_search(env=env, h_weight=h_weight)


class AStarEpsilonAgent(WeightedAStarEpsilonAgent):
    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        return super().super_search(env=env, epsilon=epsilon)
