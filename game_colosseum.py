from game import GameState

from math import *
import numpy as np
from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT
from definitions import PARTIAL_OBSERVABILITY, MCTS_ADDITIONAL_SEARCH
from definitions import SAVE_PERIOD

from record_functions import Draw_Plot, Draw_Win_Graph, Record

# Function Game_Collosseum
#  1. Get two model
#  2. Play a game with two models
#  3. Return win model's identifier, win_score

# - Assumption 1 : Each model considers its stones as "1", enemy's stone as -1
# - Assumption 2 : Models should know "PASS_MOVE" state, which placing more stone is unnecessary
def Game_Collosseum(black_model, white_model, game_count) :

    print("Enter Collosseum")

    Go_Game = GameState() # board size = 19

    white_win_prob = []
    black_win_prob = []
    
    try :
        while (True) :
            # - Get current Player
            current_player = Go_Game.get_current_player()

            # - Get action from predicted model
            if (current_player is BLACK) :
                action, win_prob = black_model.get_move(BLACK, Go_Game)
                print("Black win prob is : ", str(win_prob * 100.0), " %")
                black_win_prob.append(win_prob)
            else :
                action, win_prob = white_model.get_move(WHITE, Go_Game)
                print("White win prob is : ", str(win_prob * 100.0), " %")
                white_win_prob.append(win_prob)
            
            # - Act move & get results 
            if (Go_Game.do_move(action) is True) :
                winner, black_go, white_go = Go_Game.get_winner()

                if (winner is BLACK) :
                    print("winner : BLACK")
                    Record(str("Winner is BLACK - Game : " + str(game_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")
                else :
                    print("winner : WHITE")
                    Record(str("Winner is WHITE - Game : " + str(game_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")

                board = np.array(Go_Game.show_result())
                Draw_Plot(board, save_image=True, save_image_name=str("result/snapshot/snapshot_" + str(game_count) + ".png"))

                if (game_count % SAVE_PERIOD) is 0 :
                    Draw_Win_Graph(white_win_prob, figure_num=5, save_image_name=str("result/white_win/white_win_" + str(game_count) + ".png"))
                    Draw_Win_Graph(black_win_prob, figure_num=6, save_image_name=str("result/black_win/black_win_" + str(game_count) + ".png"))

                break

            # - Create Plot
            board = np.array(Go_Game.show_result())
            Draw_Plot(board)

        del white_win_prob
        del black_win_prob

        del Go_Game
        return winner
    
    # ILLEGAL MOVE
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
    
