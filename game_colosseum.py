from game import GameState

from math import *
import numpy as np
from matplotlib import pyplot as plt

WHITE = -1
BLACK = 1
PASS_MOVE = None

# Draw plot with Board
def Draw_Plot(board, 
              figure_num=3, 
              pause_time=0.1,
              save_image=False,
              save_image_name=None) :

    plt.figure(figure_num, clear=True)

    plt.imshow(board)

    for i in range(0, 19) :
        plt.axhline(y=i, color='black', linewidth=1)
        plt.axvline(x=i, color='black', linewidth=1)

    plt.title("Board")
    plt.colorbar()  

    plt.draw()
    plt.pause(pause_time)

    if (save_image is True) :
        plt.savefig(save_image_name)
    
    return

# Draw win probability graph
def Draw_Win_Graph(win_prob_list,
                   figure_num=5,
                   save_image_name=None) :
    
    plt.figure(figure_num, clear=True)
    plt.plot(win_prob_list) # X : [0 ... N-1]

    plt.savefig(save_image_name)

    return

# Record something into string
def Record(string,
           filename,
           append_mode=True) :
    
    if (append_mode is True) :
        file = open(filename, "a")
    else :
        file = open(filename, "w")

    file.write(str(string))
    file.close()

    return

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
                black_win_prob.append(win_prob)
            else :
                action, win_prob = white_model.get_move(WHITE, Go_Game)
                white_win_prob.append(win_prob)
            
            # - Act move & get results 
            if (Go_Game.do_move(action) is True) :
                winner = Go_Game.get_winner()

                if (winner is BLACK) :
                    print("winner : BLACK")
                    Record(str("Winner is BLACK - Game : " + str(game_count)), "result/game_results.txt")
                else :
                    print("winner : WHITE")
                    Record(str("Winner is WHITE - Game : " + str(game_count)), "result/game_results.txt")

                board = np.array(Go_Game.show_result())
                Draw_Plot(board, save_image=True, save_image_name=str("snapshot_" + str(game_count) + ".png"))

                Draw_Win_Graph(black_win_prob, figure_num=5, save_image_name=str("white_win_" + str(game_count) + ".png"))
                Draw_Win_Graph(black_win_prob, figure_num=6, save_image_name=str("black_win_" + str(game_count) + ".png"))

                break

            # - Create Plot
            board = np.array(Go_Game.show_result())
            Draw_Plot(board)

        del white_win_prob
        del black_win_prob

        del Go_Game
        return winner
    
    # ILLEGAL MOVE
    except ValueError :
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
    
