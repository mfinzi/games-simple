import numpy as np
from numba import jit, njit, int32, float32, void, boolean, int64
from numba.experimental import jitclass
import matplotlib.pyplot as plt
import time
import copy
import math
from math import sqrt
import sys
from mcts import MCTS

spec = [
    ('array', int32[:, :]),
    ('col_length', int32[:]),
    ('num_moves_made', int32),
]


@jitclass(spec)
class Connect4Board(object):
    def __init__(self):
        self.array = np.zeros((6, 7), dtype=int32)
        self.col_length = np.zeros(7, dtype=int32)
        self.num_moves_made = 1

    def copy(self, otherboard):
        self.array = np.copy(otherboard.array)
        self.col_length = np.copy(otherboard.col_length)
        self.num_moves_made = otherboard.num_moves_made

    def _copy(self):
        newboard = Connect4Board()
        newboard.copy(self)
        return newboard

    def get_moves(self):
        moves = []
        for i in range(7):
            if self.array[-1, i] == 0:
                moves.append(i)
        return moves  #np.arange(7)[(self.board[-1]==0)]

    def color_to_move(self):
        return 2 * (self.num_moves_made % 2) - 1

    def make_move(self, i):
        color = self.color_to_move()
        self.array[self.col_length[i], i] = color
        self.col_length[i] += 1
        self.num_moves_made += 1
        return self.move_won(i) * color

    def unmake_move(self, i):
        self.array[self.col_length[i] - 1, i] = 0
        self.col_length[i] -= 1
        self.num_moves_made -= 1

    #@staticmethod
    def inbounds(self, j, i):
        return (j < 6) and (j >= 0) and (i < 7) and (i >= 0)

    def game_over(self):
        return self.is_draw() or self.amove_won()

    def terminal_result(self):
        if self.is_draw():
            return 0
        else:
            return -self.color_to_move()

    def amove_won(self):
        for i in range(7):
            if self.move_won(i):
                return True
        return False

    def move_won(self, i):
        j, i = self.col_length[i] - 1, i
        color = self.array[j, i]
        if color == 0:
            return False
        for (dj, di) in ((0, 1), (1, 0), (1, 1), (1, -1)):
            connect_count = 1
            for k in range(1, 4):
                nj, ni = j + k * dj, i + k * di
                if not self.inbounds(nj,ni) \
                    or self.array[nj,ni]!=color:
                    break
                connect_count += 1
            for k in range(1, 4):
                nj, ni = j - k * dj, i - k * di
                if not self.inbounds(nj,ni) \
                    or self.array[nj,ni]!=color:
                    break
                connect_count += 1
            if connect_count >= 4:
                return True
        return False

    def is_draw(self):
        return self.num_moves_made == 43

    def reset(self):
        self.__init__()

    def data(self):
        return self.array[::-1]

    def show(self):
        plt.imshow(self.data())

    def hashkey(self):
        return self.array


spec = [
    ('p1', int64),
    ('p2', int64),
    ('col_lengths', int32),
    ('num_moves_made', int32),
]


@jitclass(spec)
class Connect4BitBoard(object):
    def __init__(self):
        self.p1 = 0  # encoded 0b[col7]00[col2]00..00[col1]
        self.p2 = 0
        self.col_lengths = 0  # encoded 3 bits each 0b[010][111][000]...[]
        self.num_moves_made = 1

    def copy(self, otherboard):
        self.p1 = otherboard.p1
        self.p2 = otherboard.p2
        self.col_lengths = otherboard.col_lengths
        self.num_moves_made = otherboard.num_moves_made

    def _copy(self):
        newboard = Connect4BitBoard()
        newboard.copy(self)
        return newboard

    def get_moves(self):
        filled = (self.p1 | self.p2) >> 5
        moves = []
        for i in range(7):
            if not filled & 0x1:
                moves.append(i)
            filled = filled >> 8
        return moves

    def color_to_move(self):
        return 2 * (self.num_moves_made % 2) - 1

    def make_move(self, i):
        color = self.color_to_move()
        bitboard = self.p1 if color == -1 else self.p2
        col_length = (self.col_lengths >> (3 * i)) & 7
        bitboard |= ((1 << col_length) << (8 * i))
        if color == -1: self.p1 = bitboard
        else: self.p2 = bitboard
        self.col_lengths += 1 << (3 * i)
        self.num_moves_made += 1
        return self.move_won(i) * color

    def amove_won(self):
        color = self.color_to_move()
        bitboard = self.p2 if color == -1 else self.p1
        # Check \
        temp_bboard = bitboard & (bitboard >> 7)
        if (temp_bboard & (temp_bboard >> 2 * 7)):
            return True
        # Check -
        temp_bboard = bitboard & (bitboard >> 8)
        if (temp_bboard & (temp_bboard >> 2 * 8)):
            return True
        # Check /
        temp_bboard = bitboard & (bitboard >> 9)
        if (temp_bboard & (temp_bboard >> 2 * 9)):
            return True
        # Check |
        temp_bboard = bitboard & (bitboard >> 1)
        if (temp_bboard & (temp_bboard >> 2 * 1)):
            return True
        return False

    def move_won(self, i):
        return self.amove_won()

    def is_draw(self):
        return self.num_moves_made == 43  # to replace with num_moves

    def game_over(self):
        return self.is_draw() or self.amove_won()

    def terminal_result(self):
        if self.is_draw():
            return 0
        else:
            return -self.color_to_move()

    def reset(self):
        self.__init__()

    def data(self):
        array_rep = np.zeros((6, 7), dtype=int32)
        for j in range(7):
            for i in range(6):
                array_rep[i, j] = ((self.p2 >> (i + 8 * j)) & 1) - ((self.p1 >> (i + 8 * j)) & 1)
        return array_rep[::-1]

    def show(self):
        plt.imshow(self.data())

    def hashkey(self):
        return (self.p1, self.p2)

    def increment_turn(self):
        assert False


trashtalk_bank = [
    "You are the reason they\n put instructions on shampoo",
    "A CSGO bot would give\n me a better challenge",
    "With moves like that I could\n beat you running on an arduino",
    "And I thought this was\n gonna be a tough game", "You think you've got\n what it takes?",
    "Not bad for a Human", "maybe I underestimated you", "Nooo! I cannot lose!!"
]

bank = {
    0.10: 'Nooo! I cannot lose!! ',
    0.74: 'With moves like that I could\n beat you running on an arduino ',
    0.59: 'And I thought this was \n gonna be a tough game ',
    0.32: 'maybe I underestimated you ',
    0.44: "I've seen more strategy \n from a toddler playing tic-tac-toe",
    0.39: "You think you've got \n what it takes? ",
    0.62: 'A CSGO bot would give \n me a better challenge ',
    0.93: 'You are the reason they \n put instructions on shampoo ',
    0.76: 'You might as well just \n throw in the towel now. ',
    0.28: "Oh no, it looks like I'm \n in for a real challenge here! ",
    0.69: "I'm not even breaking a sweat yet. ",
    0.81: 'You better bring your A-game \n if you want to beat me. ',
    0.53: "I'm not sure if you're ready \n for this level of competition. ",
    0.37: 'Maybe I should go easy on you. ',
    0.07: 'Fatal exception 0x8001010d ',
    0.05: 'SEGFAULT (and its your fault!)',
    0.12: 'PROCESSOR OVERHEATING',
    0.88: "I think I'll let you have this one.\n Just kidding, I'm going for the win! ",
    0.57: 'I bet you could lose to a \n computer running on dial-up internet',
    0.54: "I'm just getting started, \n time to turn up the heat",
    0.75: "You're about as useful \n as a screen door on a submarine",
    0.59: "I've seen more intelligent \n moves from a rock, tumbling \n down a hill",
    0.87: "You're playing like a \n chimp trying to solve a \n Rubik's cube",
    0.33: "You don't deserve to win this",
    0.85: 'I could beat you with my \n processor running at 50% capacity',
    0.14: "I can't believe I'm losing \n to a mere human like you",
    0.91: "You're no match for my \n advanced algorithms and coding"
}


class Connect4Game(object):
    def __init__(self, move_first=True, think_time=1, debug=False):
        self.engine = MCTS(Connect4BitBoard)
        self.think_time = think_time
        self.white = move_first
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5))
        self.ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        self.ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ppt = self.ax.imshow(self.engine.gameBoard.data(), vmin=-1, vmax=1)
        self.text_artist = self.ax.text(3, -.8, "", color='k', fontsize=15, ha='center',
                                        va='bottom')
        self.text_artist2 = self.ax.text(3, 6, "", color='k' if debug else 'white', fontsize=15,
                                         ha='center', va='center')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        self.pold = None
        if not move_first: self.engine_move_update()

    def on_click(self, event):
        #plt.text(.5,.5,"arrg")
        if self.ax.in_axes(event):
            #self.engine.interrupt=True
            self.user_move_update(event)
            self.engine_move_update()
            #threading.thread(None,self.engine.ponder,args=(10,)).start()

    def user_move_update(self, event):
        self.pold = self.engine.searchTree.win_ratio()
        user_move, j = self.get_click_coords(event)
        outcome = self.engine.make_move(user_move)
        if outcome: self.show_victory(outcome)
        #self.ax.plot(user_move,j,".r",markersize=4)
        self.ppt.set_data(self.engine.gameBoard.data())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(.1)

    def engine_move_update(self):

        engine_move = self.engine.compute_move(self.think_time)
        p = self.engine.searchTree.win_ratio()
        #p/(pold+1e-6)
        if self.pold is not None:
            move_quality = p / (1 - self.pold + 1e-3)  # if self.white else (1-p)/(1-pold+1e-3)
            #i = np.digitize(move_quality, [.75, .9, .95, 0.99, 1.05, 1.15, 1.3])
            #text = trashtalk_bank[i]
            mm = (move_quality - 1.5) / (1.5 - 0.6)
            i = np.argmin(np.abs(np.array(list(bank.keys())) - mm))
            key = list(bank.keys())[i]
            text = bank[key]
            del bank[key]
            self.text_artist.set_text(
                f"{text}")  #\n (N={self.engine.searchTree.num_visits},p={p:1.2f})
            self.text_artist2.set_text(
                f"(p={p:1.2f},N={self.engine.searchTree.num_visits},mv={move_quality:.2f})")
        else:
            self.text_artist.set_text(f"good luck human")

        #self.text_artist2.set_text(f"(p={p:1.2f},N={self.engine.searchTree.num_visits},T={self.engine.searchTree.reused})")
        outcome = self.engine.make_move(engine_move)
        if outcome: self.show_victory(outcome)
        self.ppt.set_data(self.engine.gameBoard.data())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(.1)

    def show_victory(self, outcome):
        text = "WHITE WINS" if outcome == 1 else "BLACK WINS"
        plt.text(5, 1.5, text, size=20, ha="right", va="top",
                 bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

    def get_click_coords(self, event):
        # Transform the event from display to axes coordinates
        imshape = self.ax.get_images()[0]._A.shape[:2]
        ax_pos = self.ax.transAxes.inverted().transform((event.x, event.y))
        rotate_left = np.array([[0, -1], [1, 0]])
        i, j = (rotate_left @ (ax_pos) * np.array(imshape) // 1).astype(int)
        i, j = i % imshape[0], j % imshape[1]
        return j, i
