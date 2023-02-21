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
    ('board', int32[:]),
    ('turn', int32),
]


@jitclass(spec)
class MancalaBoard(object):
    def __init__(self):
        self.board = np.zeros(14, dtype=int32)
        self.board[:6] += 4
        self.board[7:13] += 4
        self.turn = 1

    def copy(self, otherboard):
        self.board = np.copy(otherboard.board)
        self.turn = otherboard.turn

    def _copy(self):
        newboard = MancalaBoard()
        newboard.copy(self)
        return newboard

    def get_moves(self):
        moves = []
        player = self.turn ^ 1
        for i in range(6):
            mv = player * 7 + i
            if self.board[mv] > 0:
                moves.append(mv)
        return moves

    def color_to_move(self):
        return 2 * self.turn - 1

    def increment_turn(self):
        self.turn = self.turn ^ 1

    def make_move(self, i):
        count = self.board[i]
        self.board[i] = 0
        playerid = self.turn ^ 1  #self.num_moves_made % 2
        used = 0
        j = 0
        # for j in range(count):
        #     target = (i + j + 1) % 14
        #     self.board[target] += 1
        while used < count:
            target = (i + j + 1) % 14
            otherplayer_store = self.turn * 7 + 6
            j += 1
            if target == otherplayer_store:
                continue
            self.board[target] += 1
            used += 1
        # check of last move was not in own store
        end = (i + count) % 14
        player_store = playerid * 7 + 6
        if self.board[end] == 1 and end != 6 and end != 13 and end // 7 == playerid:
            opposite = 12 - end
            self.board[player_store] += self.board[opposite]
            self.board[opposite] = 0
        if end != player_store:
            self.increment_turn()
        return 0

    def game_over(self):
        return (self.board[6] + self.board[13]) >= 48

    def terminal_result(self):
        return np.tanh((self.board[6] - self.board[13]) / 4.)

    def reset(self):
        self.__init__()

    def data(self):
        extended = np.zeros((2, 8))
        extended[1, 1:] = self.board[:7]
        extended[0, :-1] = self.board[7:][::-1]
        return extended

    # def show(self):
    #     plt.imshow(self.data())

    def hashkey(self):
        return (self.board, self.turn)


def play_game(bot1, bot2, time1=.1, time2=None):
    game = MancalaBoard()
    game.make_move(2)
    bot1.make_move(2)
    bot2.make_move(2)
    even_move = 0
    game.make_move(even_move)
    bot1.make_move(even_move)
    bot2.make_move(even_move)
    bot1.ponder(2)
    bot2.ponder(2)
    while not game.game_over():
        if not len(game.get_moves()):
            bot1.gameBoard.increment_turn()
            bot2.gameBoard.increment_turn()
            game.increment_turn()
            continue
        if game.turn == 1:
            bot1_move = bot1.compute_move(time1)
            bot1.make_move(bot1_move)
            bot2.make_move(bot1_move)
            game.make_move(bot1_move)
            print("bot1 move", bot1_move)
        else:
            bot2_move = bot2.compute_move(time2 or time1)
            bot1.make_move(bot2_move)
            bot2.make_move(bot2_move)
            game.make_move(bot2_move)
            print("bot2 move", bot2_move)
        # print(f'bot1 odds {bot1.searchTree.win_ratio():.3f},N1:{bot1.searchTree.num_visits} ' +
        #       f'bot2 odds {bot2.searchTree.win_ratio():.3f},N2:{bot2.searchTree.num_visits}')
        print(board2data(game.board).astype(int))
    return game.terminal_result()


def board2data(board):
    extended = np.zeros((2, 8))
    extended[1, 1:] = board[:7]
    extended[0, :-1] = board[7:][::-1]
    return extended


trashtalk_bank = [
    "You are the reason they\n put instructions on shampoo",
    "A CSGO bot would give\n me a better challenge",
    "With moves like that I could\n beat you running on an arduino",
    "And I thought this was\n gonna be a tough game", "You think you've got\n what it takes?",
    "Not bad for a Human", "maybe I underestimated you", "Nooo! I cannot lose!!"
]

prompt = """
The following are some very humorous trashtalk lines to use for the bot to say when it has a given chance of winning.

0.15| Nooo! I cannot lose!! \n\n
0.74| With moves like that I could\n beat you running on an arduino \n\n
0.56| And I thought this was\n gonna be a tough game \n\n
0.32| maybe I underestimated you \n\n
0.44| Not bad for a Human" \n\n
0.39| You think you've got\n what it takes? \n\n
0.62| A CSGO bot would give\n me a better challenge \n\n
0.91| You are the reason they\n put instructions on shampoo \n\n
"""

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

# openai.Completion.create(engine=self.engine,prompt=prompt, logprobs=5, max_tokens=1, temperature=0.7)
#         policy_logprobs = policy_completion.choices[0]


class MancalaGame(object):
    def __init__(self, move_first=True, think_time=1, debug=False):
        self.engine = MCTS(MancalaBoard)
        self.think_time = think_time
        self.white = move_first
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 3))
        self.ax.grid(which='minor', color='k', linestyle='-', linewidth=2)
        self.ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        data = self.engine.gameBoard.data()
        #range = np.arange(14)
        #data = np.zeros((2, 8))
        #data[1, 1:] = range[:7]
        #data[0, :-1] = range[7:][::-1]
        self.ppt = self.ax.imshow(data, vmin=0, vmax=6, cmap='Blues')
        self.numbers = [
            self.ax.text(i, j, int(count), ha='center', va='center', fontsize=15)
            for (j, i), count in np.ndenumerate(data)
        ]
        self.text_artist = self.ax.text(1.6, -.8, "good luck human", color='k', fontsize=15,
                                        ha='center', va='bottom')
        self.text_artist2 = self.ax.text(7., -.8, "", color='k' if debug else 'white', fontsize=15,
                                         ha='center', va='center')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        self.pold = None
        if not move_first: self.engine_move_update()
        self.prev_move = False

    def on_click(self, event):
        #plt.text(.5,.5,"arrg")
        if self.ax.in_axes(event):
            #self.engine.interrupt=True
            mv = self.get_move(event)
            moves = self.engine.gameBoard.get_moves()
            if not (mv in moves or len(moves) == 0):
                return
            if len(moves): self.user_move_update(mv)
            else:
                self.engine.gameBoard.increment_turn()
            engine_turn = lambda: not (self.white ^ self.engine.gameBoard.turn ^ 1
                                       ) and not self.engine.gameBoard.game_over()
            while engine_turn():
                if len(self.engine.gameBoard.get_moves()):
                    self.text_artist.set_text(f"Thinking...")
                    self.engine_move_update()
                    time.sleep(0.5)
                else:
                    return
            #threading.thread(None,self.engine.ponder,args=(10,)).start()
    def get_move(self, event):
        i, j = self.get_click_coords(event)
        user_move = i - 1 if j == 1 else 13 - i
        #print(user_move)
        #self.text_artist2.set_text(f"ij: {(i,j)} Move: {user_move}")
        return user_move

    def user_move_update(self, user_move):
        self.pold = self.engine.searchTree.win_ratio()
        self.update_display(user_move)
        self.engine.make_move(user_move)
        self.render_data(self.engine.gameBoard.board)
        self.prev_move = True
        #self.ax.plot(user_move,j,".r",markersize=4)

    def render_data(self, array):
        data = board2data(array)
        self.ppt.set_data(data)
        for (k, l), c in np.ndenumerate(data):
            self.numbers[k * 8 + l].set_text(int(c))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(.05)

    def sequential_display_move(self, i):
        board = self.engine.gameBoard
        playerid = board.turn ^ 1
        array = np.copy(board.board)
        count = array[i]
        array[i] = 0
        used = 0
        j = 0
        while used < count:
            target = (i + j + 1) % 14
            otherplayer_store = board.turn * 7 + 6
            j += 1
            self.render_data(array)
            if target == otherplayer_store:
                continue
            array[target] += 1
            used += 1
        self.render_data(array)

        end = (i + count) % 14
        player_store = playerid * 7 + 6
        if array[end] == 1 and end != 6 and end != 13 and end // 7 == playerid:
            opposite = 12 - end
            array[player_store] += array[opposite]
            array[opposite] = 0
            self.render_data(array)

    def update_display(self, move):
        self.sequential_display_move(move)
        #self.render_data(self.engine.gameBoard.board)
        if self.engine.gameBoard.game_over():
            outcome = self.engine.gameBoard.terminal_result()
            self.show_victory(outcome)
        #self.ppt.set_data(self.engine.gameBoard.data())
        # for (j, i), count in np.ndenumerate(self.engine.gameBoard.data()):
        #     self.numbers[j * 8 + i].set_text(count)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # time.sleep(.01)

    def engine_move_update(self):
        engine_move = self.engine.compute_move(self.think_time)
        self.update_display(engine_move)
        self.engine.make_move(engine_move)
        self.render_data(self.engine.gameBoard.board)
        p = self.engine.searchTree.win_ratio()
        #p/(pold+1e-6)
        #if self.pold is not None:
        move_quality = p  #/ (1 - self.pold + 1e-3)  # if self.white else (1-p)/(1-pold+1e-3)
        #i = np.digitize(move_quality, [.2, .35, .45, .52, .70, .80, .90][::-1])

        #i = np.digitize(move_quality, [.75, .9, .95, 0.99, 1.05, 1.15, 1.3])
        #text = trashtalk_bank[i]
        # find closest element in trashtalk bank
        if (self.white ^ self.engine.gameBoard.turn ^ 1) and bank:
            i = np.argmin(np.abs(np.array(list(bank.keys())) - move_quality))
            key = list(bank.keys())[i]
            text = bank[key]
            del bank[key]
            #if self.i_last != i:
            self.text_artist.set_text(
                f"{text}")  #\n (N={self.engine.searchTree.num_visits},p={p:1.2f})
        #self.i_last = i
        self.text_artist2.set_text(
            f"(p={p:1.2f},N={self.engine.searchTree.num_visits})")  #,mv={move_quality:.2f})")
        #else:
        #self.pold = p
        # self.text_artist.set_text(f"good luck human")

        #self.text_artist2.set_text(f"(p={p:1.2f},N={self.engine.searchTree.num_visits},T={self.engine.searchTree.reused})")
        # self.engine.make_move(engine_move)
        # if self.engine.gameBoard.game_over():
        #     outcome = self.engine.gameBoard.terminal_result()
        #     self.show_victory(outcome)
        # self.ppt.set_data(self.engine.gameBoard.data())
        # for (j, i), count in np.ndenumerate(self.engine.gameBoard.data()):
        #     self.numbers[j * 8 + i].set_text(count)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # time.sleep(.01)

    def show_victory(self, outcome):
        text = "WHITE WINS" if outcome > 0 else "BLACK WINS"
        plt.text(1.6, 0, text, fontsize=20, ha='center', va='bottom',
                 bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

    def get_click_coords(self, event):
        # Transform the event from display to axes coordinates
        imshape = self.ax.get_images()[0]._A.shape[:2]
        ax_pos = self.ax.transAxes.inverted().transform((event.x, event.y))
        rotate_left = np.array([[0, -1], [1, 0]])
        i, j = (rotate_left @ (ax_pos) * np.array(imshape) // 1).astype(int)
        i, j = i % imshape[0], j % imshape[1]
        return j, i
