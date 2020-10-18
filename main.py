import model
import game_colosseum

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import SAVE_PERIOD

from record_functions import Draw_Plot, Draw_Win_Graph, Record

# Create 2 -> 1 model

def main_init_model() :
    # Create 2 model
    black_model = model.RL_Model()
    white_model = model.RL_Model()

    black_model.set_team(BLACK)
    white_model.set_team(WHITE)

    # Restore models from "restore_point"
    train_count_black = black_model.restore_model(restore_point=None)
    train_count_white = white_model.restore_model(restore_point=None)

    black_model.hang_opposite_model(white_model)
    white_model.hang_opposite_model(black_model)

    train_count = min(train_count_black, train_count_white)

    return train_count, black_model, white_model


if __name__ == "__main__" :
    train_count, black_model, white_model = main_init_model()

    black_win = 0
    white_win = 0

    while (True) :  
        print("Train : ", train_count)
        train_count += 1

        # Prepare game
        # set_team function includes initializing model.
        black_model.set_team(BLACK)
        white_model.set_team(WHITE)

        # Play game
        winner, board, _ , black_win_prob, white_win_prob, black_go, white_go = game_colosseum.Game_Collosseum(black_model, white_model, train_count)

        # Record History
        if (winner is BLACK) :
            Record(str("Winner is BLACK - Game : " + str(train_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")
        else :
            Record(str("Winner is WHITE - Game : " + str(train_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")

        # Draw Plot and Winning rate graph, and save it onto storage.
        Draw_Plot(board, save_image=True, save_image_name=str("result/snapshot/snapshot_" + str(train_count) + ".png"))
        Draw_Win_Graph(white_win_prob, figure_num=5, save_image_name=str("result/white_win/white_win_" + str(train_count) + ".png"))
        Draw_Win_Graph(black_win_prob, figure_num=6, save_image_name=str("result/black_win/black_win_" + str(train_count) + ".png"))

        if (winner is BLACK) :
            black_win += 1
        else : # winner is WHITE
            white_win += 1
    
        Record(str(" BLACK WIN : "  + str(black_win) + " | WHITE WIN : " + str(white_win) + "\n"),  "result/win_count.txt")

        del black_win_prob, board, white_win_prob

        print("winner value is ", str(winner))

        # Train
        black_model.train_func(winner)
        white_model.train_func(winner)

        # Save model
        if (train_count % SAVE_PERIOD) is 0 :
            black_model.save_model(train_count)
            white_model.save_model(train_count)
    
