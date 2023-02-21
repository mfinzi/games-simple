import numpy as np
from numba import jit, njit, int32, float32, void, boolean, typed
import time
import math
from math import sqrt
import copy


def hashkey(board):
    key = board.hashkey()
    return (key[0].tostring(), key[1])
    #return key if isinstance(key, tuple) else key.tostring()


@njit
def ucb(num_wins, num_visits, sqrtlogN, puct):
    win_rate = num_wins / num_visits
    explore_bonus = sqrtlogN / math.sqrt(num_visits)
    return win_rate + puct * explore_bonus

    #return (board.p1,board.p2)#hash(board.array.tostring())


class SearchNode(object):

    transposition_table = {}
    reused = 0
    #sqrtlogN = 0
    @classmethod
    def reset(cls):
        #cls.sqrtlogN =0
        cls.transposition_table = {}
        cls.reused = 0

    __slots__ = ('moves', 'children', 'unvisited', 'num_visits', 'num_wins', 'sqrtlogN', 'puct',
                 'color')

    def __init__(self, puct=1.5):  #move, number of visits, wins
        self.moves = []
        self.children = []
        self.unvisited = None
        self.num_visits = 0
        self.num_wins = 0
        self.sqrtlogN = 0
        self.puct = puct
        self.color = 0

    @staticmethod
    @njit
    def rollout(board):
        while True:
            if board.game_over():
                return board.terminal_result()
            moves = board.get_moves()
            if len(moves) == 0:
                board.increment_turn()
                continue
            move = moves[np.random.randint(len(moves))]
            outcome = board.make_move(move)
            if outcome: return outcome

    def win_ratio(self):
        if self.num_visits == 0: return None
        win_rate = self.num_wins / self.num_visits
        return win_rate

    def best_child_id(self, final=False):
        # fix final
        if final: key = lambda i: self.children[i].win_ratio()
        else:
            # wins = (c.num_wins for c in self.children)
            # visits = (c.num_visits for c in self.children)
            # return self.best_id(wins, visits, self.puct, self.sqrtlogN)
            # return max((ucb(c.num_wins, c.num_visits, self.sqrtlogN, self.puct), i)
            #            for i, c in enumerate(self.children))[1]
            key = lambda i: self.ucb(self.children[i].num_wins, self.children[i].num_visits, self.
                                     sqrtlogN, self.puct)
        return max(range(len(self.children)), key=key)

    # @staticmethod
    # @njit
    # def best_id(wins, visits, puct, sqrtLogN):
    #     m = -1
    #     imax = -1
    #     for i, (win, visit) in enumerate(zip(wins, visits)):
    #         val = ucb(win, visit, sqrtLogN, puct)
    #         if val > m:
    #             m = val
    #             imax = i
    #     # _, imax = max((SearchNode.ucb(c.num_wins, c.num_visits, sqrtlogN, puct), i)
    #     #               for i, c in enumerate(children))
    #     return imax

    @staticmethod
    @njit
    def ucb(num_wins, num_visits, sqrtlogN, puct):
        win_rate = num_wins / num_visits
        explore_bonus = sqrtlogN / math.sqrt(num_visits)
        return win_rate + puct * explore_bonus

    def terminal_outcome(self, board):
        if board.game_over():
            return board.terminal_result()
        else:
            return None  # Nothing

    def update_path(self, board):
        #board_color = -1 * board.color_to_move()

        # leaf node, either terminal or unvisited
        if len(self.moves) == 0:
            terminal_outcome = self.terminal_outcome(board)
            if terminal_outcome is not None: outcome = terminal_outcome
            else:
                self.moves = board.get_moves()
                self.children = [None] * len(self.moves)
                self.unvisited = np.random.permutation(len(self.moves))
                outcome = 0  #self.rollout(board)  #
                for _ in range(3):
                    newboard = board._copy()
                    outcome += self.rollout(newboard)
                outcome /= 3

        # Node has not been fully expanded
        elif len(self.unvisited):  #np.any(self.unvisited):
            child = self.expand_unvisited(board)
            outcome = child.update_path(board)

        # Node has been fully expanded and we use the (ucb) policy
        else:
            m = self.best_child_id()
            board.make_move(self.moves[m])
            outcome = self.children[m].update_path(board)

        self.update_statistics(outcome)
        return outcome

    def update_statistics(self, outcome):
        self.num_visits += 1
        self.num_wins += 0.5 * (1. + self.color * outcome)
        self.sqrtlogN = np.sqrt(2 * np.log(self.num_visits))
        #self.color = color

    def expand_unvisited(self, board):
        """ Finds an unvisited child node, adds/checks transpose table,
         makes move, returns child"""
        m = self.unvisited[-1]
        self.unvisited = self.unvisited[:-1]
        child = self.children[m] = SearchNode(self.puct)
        move = self.moves[m]
        color = board.color_to_move()  # color corresponds to action from prev node
        board.make_move(move)
        child.color = color
        # key = hashkey(board)
        # if key in SearchNode.transposition_table:
        #     SearchNode.reused += 1
        #     self.children[m] = SearchNode.transposition_table[key]
        #     child = self.children[m]
        # else:
        #     SearchNode.transposition_table[key] = child
        return child


class MCTS(object):
    def __init__(self, boardType, puct=1.5):
        self.boardType = boardType
        self.gameBoard = boardType()
        self.searchTree = SearchNode(puct)
        self.puct = puct
        # self.interrupt=False
        SearchNode.reset()

    def ponder(self, think_time):
        start_time = time.time()
        new_board = self.boardType()
        while time.time() - start_time < think_time:
            new_board.copy(self.gameBoard)
            self.searchTree.update_path(new_board)
            #SearchNode.sqrtlogN = np.sqrt(2*np.log(self.searchTree.num_visits))

    def compute_move(self, think_time):
        self.ponder(think_time)
        m = self.searchTree.best_child_id(True)
        return self.searchTree.moves[m]

    def make_move(self, move):
        legal_moves = np.array(self.gameBoard.get_moves())
        assert move in legal_moves
        # find the associated child
        m = np.nonzero(move == legal_moves)[0][0]
        if self.searchTree.children:
            child = self.searchTree.children[m]
        else:
            child = SearchNode(self.puct)
        # Update the search tree
        # Discard the tree and table for now
        self.searchTree = child
        #Node.num_rollouts = self.searchTree.num_visits
        SearchNode.reset()
        #self.searchTree = Node(move)
        outcome = self.gameBoard.make_move(move)
        return outcome
