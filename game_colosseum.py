from game import GameState

from math import *
import numpy as np
from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE
from definitions import PARTIAL_OBSERVABILITY, ADDITIONAL_SEARCH_COUNT_DURING_MCTS
from definitions import SAVE_PERIOD

from record_functions import Draw_Plot

from game_hub import *

# Function Game_Collosseum
#  1. Get two model
#  2. Play a game with two models
#     [1] Each model decides proper action from current's game state at model's turn.
#     [2] Record model's results into history buffer for each move.
#     [-] From model's history buffer, draw plot. (main.py) 
#  3. Return win model's identifier, win_score, history informations

# - Assumption 1 : Each model considers BLACK is 1, WHITE is -1, and models know each one's COLOR.
# - Assumption 2 : Models should know "PASS_MOVE" state, which placing more stones is unnecessary.
#                  In that case, model itself should skip recording informations(win_prob) at training sequence.
def Game_Collosseum(black_model, white_model, game_count, add_sample=True, debug_mode=0) : # Debug mode 0 means it will not use debugging

    print("Enter Collosseum")

    Go_Game = GameState() # board size = 19

    # History buffers
    white_win_prob_history_buffer = []
    black_win_prob_history_buffer = []
    move_history_buffer = [] # First player is black

    black_inputs_array = []
    white_inputs_array = []

    black_spot_prob = []
    white_spot_prob = []
    
    try :
        while (True) :
            current_player = Go_Game.get_current_player()

            # - Calculate Proper action
            # - Append prob_history
            if (current_player is BLACK) :
                action, win_prob, _ , board_input, spot_prob = black_model.get_move(BLACK, Go_Game, game_count, debug_mode=debug_mode)
                black_win_prob_history_buffer.append(win_prob)
                black_inputs_array.append(board_input)
                black_spot_prob.append(spot_prob)

                print("Black win prob is : ", str(win_prob), " , ", str(win_prob), " value (-1 ~ 1)")
            else :
                action, win_prob, _ , board_input, spot_prob = white_model.get_move(WHITE, Go_Game, game_count, debug_mode=debug_mode)
                white_win_prob_history_buffer.append(win_prob)
                white_inputs_array.append(board_input)
                white_spot_prob.append(spot_prob)

                print("White win prob is : ", str(win_prob), " , ", str(1 - win_prob), " value (-1 ~ 1)")

            # - Append move_history
            move_history_buffer.append(action)
            
            # - Act move & get results 
            # - Break while if game is finished
            if (set_move(Go_Game, action) is True) :
                winner, black_score, white_score = Go_Game.get_winner()

                board = np.array(Go_Game.show_result())
                break

            # - Draw game's current state(Board)
            board = np.array(Go_Game.show_result())
            Draw_Plot(board)

        # - Add sample
        if (add_sample is True) :
            black_model.add_samples(black_inputs_array, black_spot_prob)
            white_model.add_samples(white_inputs_array, white_spot_prob)

        # - End of collosseum
        del Go_Game
        return winner, board, move_history_buffer, black_win_prob_history_buffer, white_win_prob_history_buffer, black_score, white_score
    
    # Exception (Error handling)
    except KeyboardInterrupt :
        return

# Function Games_Collosseum
# Play Game n times, and do not add samples.
# Return Black Win / White Win counts
def Games_Collosseum(black_model, white_model, game_count, n_games) : # Debug mode 0 means it will not use debugging
    black_wins = 0
    white_wins = 0

    for _ in range(0, int(n_games)) :
        black_model.initialize_model()
        white_model.initialize_model()

        winner, _ , _ , _ , _ , _ , _ = Game_Collosseum(black_model, white_model, game_count, 
                                                add_sample=False, debug_mode=0)

        if (winner == BLACK) :
            black_wins += 1
        elif (winner == WHITE) :
            white_wins += 1

    return black_wins, white_wins



# Function Game_Collosseum_Onemove
#  1. Get Game engine, two model
#  2. Play game with Engine, act only one times.
#  3. If game is end, return (game_end == TRUE) and winner.
#     If game is not end, return (game_end == FALSE) and None.

# - Assumption 1 : Each model considers BLACK is 1, WHITE is -1, and each model knows its own COLOR.
# - Assumption 2 : Models should know "PASS_MOVE" state, which placing more stones is unnecessary.
#                  In that case, model itself should skip recording informations(win_prob) at training sequence.
def Game_Collosseum_Onemove(GameEngine, black_model, white_model, game_count, debug_mode=0) : # Debug mode 0 means it will not use debugging

    Go_Game = GameEngine
    current_player = GameEngine.get_current_player()

    # - From model, calculate next move with 
    if (current_player is BLACK) :
        action, win_prob, move_stack, _ , _ = black_model.get_move(BLACK, Go_Game, game_count, debug_mode=debug_mode)
        print("Black win prob is : ", str(win_prob), " , ", str(win_prob), " value (-1 ~ 1)")

    else :
        action, win_prob, move_stack, _ , _  = white_model.get_move(WHITE, Go_Game, game_count, debug_mode=debug_mode)
        print("White win prob is : ", str(win_prob), " , ", str(1 - win_prob), " value (-1 ~ 1)")

    # - Act move & get results 
    if (set_move(Go_Game, action) is True) :
        game_end = True
        winner, _ , _ = Go_Game.get_winner()
    else :
        game_end = False
        winner = None

    return game_end, winner, GameEngine, action, move_stack
