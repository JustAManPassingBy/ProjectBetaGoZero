from game import GameState

from math import *
import numpy as np
from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT
from definitions import PARTIAL_OBSERVABILITY, MCTS_ADDITIONAL_SEARCH
from definitions import SAVE_PERIOD

from record_functions import Draw_Plot

# Function Game_Collosseum
#  1. Get two model
#  2. Play a game with two models
#  3. Return win model's identifier, win_score, history informations

# - Assumption 1 : Each model considers BLACK is 1, WHITE is -1, and each model knows its own COLOR.
# - Assumption 2 : Models should know "PASS_MOVE" state, which placing more stones is unnecessary.
#                  In that case, model itself should skip recording informations(win_prob) at training sequence.
def Game_Collosseum(black_model, white_model, game_count) :

    print("Enter Collosseum")

    Go_Game = GameState() # board size = 19

    # History informations
    white_win_prob = []
    black_win_prob = []
    move_history = [] # Start from black
    
    try :
        while (True) :
            # - Get current Player
            current_player = Go_Game.get_current_player()

            # - Get action from predicted model
            # - Append prob_history
            if (current_player is BLACK) :
                action, win_prob = black_model.get_move(BLACK, Go_Game)
                print("Black win prob is : ", str(win_prob), " value (-1 ~ 1)")
                black_win_prob.append(win_prob)
            else :
                action, win_prob = white_model.get_move(WHITE, Go_Game)
                print("White win prob is : ", str(win_prob), " value (-1 ~ 1)")
                white_win_prob.append(win_prob)

            # - Append move_history
            move_history.append(action)
            
            # - Act move & get results 
            # - Break while if game is finished
            if (Go_Game.do_move(action) is True) :
                winner, black_go, white_go = Go_Game.get_winner()

                board = np.array(Go_Game.show_result())
                break

            # - Create Plot
            board = np.array(Go_Game.show_result())
            Draw_Plot(board)

        # - End of collosseum
        del Go_Game
        return winner, board, move_history, black_win_prob, white_win_prob, black_go, white_go
    
    # Exception (Error handling)
    except KeyboardInterrupt :
        board = np.array(Go_Game.show_result())
        
        '''
        Draw_Plot(board)
        '''

        print(Go_Game.show_result())
        
        # - Get current Player
        current_player = Go_Game.get_current_player()
        print("Illegal move for user : ", current_player)

        winner = current_player * -1

        a = 0
        while(True) :
            # infinite loop
            a += 1
        
        del Go_Game
        return winner
    
