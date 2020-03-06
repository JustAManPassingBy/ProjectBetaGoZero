import model
import game_colosseum

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import SAVE_PERIOD

from record_functions import Draw_Plot, Draw_Win_Graph, Record

# Create 2 model
black_model = model.RL_Model()
white_model = model.RL_Model()

black_model.set_team(BLACK)
white_model.set_team(WHITE)

black_model.hang_opposite_model(white_model)
white_model.hang_opposite_model(black_model)

train_black = black_model.restore_model(restore_train_counts=None)
train_white = white_model.restore_model(restore_train_counts=None)

train_count = min(train_black, train_white)

black_win = 0
white_win = 0

while (True) :  
    # Train count increases
    train_count += 1

    # Prepare game
    black_model.set_team(BLACK)
    white_model.set_team(WHITE)

    # Play game
    winner, board, _ , black_win_prob, white_win_prob, black_go, white_go = game_colosseum.Game_Collosseum(black_model, white_model, train_count)

    # Record History
    if (winner is BLACK) :
        Record(str("Winner is BLACK - Game : " + str(train_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")
    else :
        Record(str("Winner is WHITE - Game : " + str(train_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "result/game_results.txt")

    Draw_Plot(board, save_image=True, save_image_name=str("result/snapshot/snapshot_" + str(train_count) + ".png"))
    Draw_Win_Graph(white_win_prob, figure_num=5, save_image_name=str("result/white_win/white_win_" + str(train_count) + ".png"))
    Draw_Win_Graph(black_win_prob, figure_num=6, save_image_name=str("result/black_win/black_win_" + str(train_count) + ".png"))

    if (winner is BLACK) :
        black_win += 1
    else :
        white_win += 1
    
    Record(str(" BLACK WIN : "  + str(black_win) + " | WHITE WIN : " + str(white_win) + "\n"),  "result/win_count.txt")


    del black_win_prob, board, white_win_prob

    # Train
    black_model.train_func(winner)
    white_model.train_func(winner)

    # Save model
    if (train_count % SAVE_PERIOD) is 0 :
        black_model.save_model(train_count)
        white_model.save_model(train_count)
    
