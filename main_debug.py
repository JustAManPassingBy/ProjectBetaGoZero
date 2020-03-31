import model
import game_colosseum
import numpy as np


from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import SAVE_PERIOD

from record_functions import Draw_Plot, Draw_Win_Graph, Record
from game import GameState

# Create 2 model
black_model = model.RL_Model()
white_model = model.RL_Model()

black_model.set_team(BLACK)
white_model.set_team(WHITE)

black_model.hang_opposite_model(white_model)
white_model.hang_opposite_model(black_model)

# Restore models from "restore_point"
train_black = black_model.restore_model(restore_point=None)
train_white = white_model.restore_model(restore_point=None)

train_count = min(train_black, train_white)

black_win = 0
white_win = 0

while (True) :  
    # Train count increases
    train_count += 1

    # Prepare game
    black_model.set_team(BLACK)
    white_model.set_team(WHITE)

    ## PLAY_GAME_REGION
    Go_Game = GameState() # board size = 19

    # History buffers
    white_win_prob_history_buffer = []
    black_win_prob_history_buffer = []
    move_history_history_buffer = [] # Start from black
    
    move_count = 0

    while (True) :
        current_player = Go_Game.get_current_player()

        # - Calculate Proper action
        # - Append prob_history

        if (move_count == 30) or (move_count == 0) or (move_count == 1) or (move_count == 31) :
            if (current_player is BLACK) :
                action, win_prob = black_model.get_move_debug(BLACK, Go_Game, debug_mode=1)
                black_win_prob_history_buffer.append(win_prob)

                print("Black win prob is : ", str(win_prob), " value (-1 ~ 1)")
            else :
                action, win_prob = white_model.get_move_debug(WHITE, Go_Game, debug_mode=1)
                white_win_prob_history_buffer.append(win_prob)

                print("White win prob is : ", str(win_prob), " value (-1 ~ 1)")

        else : 
            if (current_player is BLACK) :
                action, win_prob = black_model.get_move(BLACK, Go_Game)
                black_win_prob_history_buffer.append(win_prob)

                print("Black win prob is : ", str(win_prob), " value (-1 ~ 1)")
            else :
                action, win_prob = white_model.get_move(WHITE, Go_Game)
                white_win_prob_history_buffer.append(win_prob)

                print("White win prob is : ", str(win_prob), " value (-1 ~ 1)")


        # - Append move_history
        move_history_history_buffer.append(action)
            
        # - Act move & get results 
        # - Break while if game is finished
        if (Go_Game.do_move(action) is True) :
            winner, black_go, white_go = Go_Game.get_winner()

            board = np.array(Go_Game.show_result())
            break

        # - Draw game's current state(Board)
        board = np.array(Go_Game.show_result())
        Draw_Plot(board)
        move_count += 1

    # - End of collosseum
    del Go_Game

    # Record History
    if (winner is BLACK) :
        Record(str("Winner is BLACK - Game : " + str(train_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")
    else :
        Record(str("Winner is WHITE - Game : " + str(train_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")

    Draw_Plot(board, save_image=True, save_image_name=str("result/snapshot/snapshot_" + str(train_count) + ".png"))
    Draw_Win_Graph(black_win_prob_history_buffer, figure_num=5, save_image_name=str("result/white_win/white_win_" + str(train_count) + ".png"))
    Draw_Win_Graph(black_win_prob_history_buffer, figure_num=6, save_image_name=str("result/black_win/black_win_" + str(train_count) + ".png"))

    if (winner is BLACK) :
        black_win += 1
    else :
        white_win += 1
    
    Record(str(" BLACK WIN : "  + str(black_win) + " | WHITE WIN : " + str(white_win) + "\n"),  "result/win_count.txt")


    del black_win_prob_history_buffer, board, white_win_prob_history_buffer

    print("winner value is ", str(winner))

    # Train
    black_model.train_func(winner)
    white_model.train_func(winner)

    # Save model
    if (train_count % SAVE_PERIOD) is 0 :
        black_model.save_model(train_count)
        white_model.save_model(train_count)
    
