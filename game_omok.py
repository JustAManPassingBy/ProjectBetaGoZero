# Define only rules about omok
import numpy as np
import copy

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE
from definitions import PARTIAL_OBSERVABILITY, NUM_STONES_TO_END_GAME, NUM_MOVES_TO_END_GAME

class OmokGame(object):

    def __init__(self, size=BOARD_SIZE):
        self._init_boards(size)

    def _init_boards(self, size=BOARD_SIZE) :
        self.board = np.zeros((size, size), dtype=int)
        self.board.fill(EMPTY)
        self.size = size
        self.current_player = BLACK

        # Latest moves
        self.latest_boards = np.zeros((PARTIAL_OBSERVABILITY, size, size), dtype=int)

        self.num_stones = 0
        self.move_count = 0

        self.winner = EMPTY

    # Check following items
    # valid : current player's stone
    # Empty : empty location of specific lane (EMPTY)
    # Block : enemy player's stone
    # Skip  : Skip block
    def __search_valid_three(self, x, y, dirx, diry, player) :
        line = []

        for i in range(-4, 5, 1) :
            search_x = x + dirx * i
            search_y = y + diry * 1

            if self._on_board(search_x, search_y) is False :
                line.append("S")
                continue

            if (self.board[search_y][search_x] is player) :
                line.append("V")
            elif (self.board[search_y][search_x] is EMPTY) :
                line.append("E")
            else :
                line.append("B")

        if (line[4] is EMPTY) :
            line[4] = "V"

        # 1. Check center-included(line[4]) three sequence "Valid" 
        # 2. And if (1) is true, check these stone is not blocked
        #  - (E) (V) (V) (V) (E) 
        for start in range(2, 5, 1) : # 2 ~ 4
            if (line[start] is "V") and (line[start + 1] is "V") and (line[start + 2] is "V") :
                if (line[start - 1] is "E") and (line[start + 3] is "E") :
                    return True

        # 1. Check center-included(line[4]) three "Valid" and one "Empty" at four sequencial spot
        # 2. And if (1) is true, check these stone is not blocked
        # - (E) (V) (E) (V) (V) (E)
        for start in range(1, 6, 1) : # 1 ~ 5
            num_valid = 0
            num_empty = 0
            num_block = 0

            for idx in range(0, 4, 1) : # (+0 ~ +3)
                if (line[start + idx] is "V") :
                    num_valid += 1
                elif (line[start + idx] is "E") :
                    num_empty += 1
                else :
                    num_block += 1

            if (num_valid is 3) and (num_empty is 1) :
                if (line[start - 1] is "S") and (line[start + 4] is "S") :
                    return True

        # - If not, return false
        return False

    def _on_board(self, position):
        (x, y) = position
        return x >= 0 and y >= 0 and x < self.size and y < self.size

    # action : move (x, y)
    # dir : searching direction
    # current_player : BLACK / WHITE
    def _check_valid_three(self, action, dir, current_player) :
        if (current_player is EMPTY) :
            print("Error : _check_valid_three - current player is empty")

        (x, y) = action
        (dirx, diry) = dir

        return self.__search_valid_three(x, y, dirx, diry, current_player)
    
    # Check last spot is win.
    # return False if not, return True if end
    def _check_win(self, action, dir, current_player) :
        if (current_player is EMPTY) :
            print("Error : _check_win - current player is empty")

        (x, y) = action
        (dirx, diry) = dir

        for i in range(-2, 3, 1) :
            search_x = x + dirx * i
            search_y = y + diry * 1

            if (self._on_board(search_x, search_y) is False) :
                continue

            if (self.board[search_y][search_x] is not current_player) :
                return False

        return True

    def show_latest_boards(self) :
        # deepcopy latest boards
        latest_boards = copy.deepcopy(self.latest_boards)

        # latest board normalization
        return (latest_boards / 2) + 0.5

    # Check whether game is done
    def is_done(self, offset=0) :
        if (self.winner is not EMPTY) :
            return True

        return ((self.num_stones > (NUM_STONES_TO_END_GAME + offset)) or (self.move_count > NUM_MOVES_TO_END_GAME))

    # Check whether specific action's current player is win
    def is_win(self, action, current_player) :
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dir in directions :
            if (self._check_win(action, dir, current_player) is True) :
                return True

        return False

    # Check whether specific spot is double three
    def double_three(self, action, current_player) :
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        valid_three = 0

        for dir in directions :
            if (self._check_valid_three(action, dir, current_player) is True) :
                vaild_three += 1

        return (valid_three >= 2)

    # Copy of the game
    def copy(self):
        other = OmokGame(self.size)
        other.board = self.board.copy()
        other.current_player = self.current_player

        other.latest_boards = self.latest_boards.copy()
        other.num_stones = self.num_stones
        other.move_count = self.move_count   

        return other

    def is_legal(self, action, current_player):
        if (action is None) :
            return False

        (x, y) = action

        if (self.board[y][x] is not EMPTY) :
            return False

        if self.double_three(action, current_player) is True :
            return False

        return True

    def get_legal_moves(self, current_player):
        legal_moves = []

        for x in range(0, self.size) :
            for y in range(0, self.size) :
                action = (x, y)

                if (self.is_legal(action, current_player) is True) :
                    legal_moves.append(action)

        return legal_moves

    def get_winner(self):
        return self.winner

    def get_current_player(self):
        return self.current_player

    def reset_board(self) :
        self._init_boards()

    def do_move(self, action, color=None):
        color = color or self.current_player

        if self.is_legal(action, color):
            (x, y) = action

            self.board[y][x] = color

            # Update Board
            self.latest_boards = self.latest_boards[1:]
            self.latest_boards = np.vstack((self.latest_boards, self.board.reshape(1, self.size, self.size)))

            if (self.is_win(action, color) is True) :
                self.winner = color

            self.current_player = -1 * color

            self.num_stones += 1
            self.move_count += 1
        else :
            raise ValueError()

        return self.is_done()

    
class IllegalMove(Exception):
    pass
