import model
import player_model
import game_colosseum

from definitions import BLACK, WHITE

from record_functions import Draw_Plot, Draw_Win_Graph, Record

# Setting PC
black_model = model.RL_Model()
white_model = model.RL_Model()

black_model.set_team(BLACK)
white_model.set_team(WHITE)

black_model.hang_opposite_model(white_model)
white_model.hang_opposite_model(black_model)

train_black = black_model.restore_model(restore_train_counts=None)
train_white = white_model.restore_model(restore_train_counts=None)

your_model = player_model.PLAYER_Model()
#your_model_two = player_model.PLAYER_Model()

team_string = input("Select Team (BLACK/WHITE) : ").upper()
if (team_string == "BLACK") :
    print("Select team BLACK")
    team = BLACK
else :
    print("Select team WHITE")
    teim = WHITE

play_count = 0

black_win = 0
white_win = 0

while (True) :  
    # Play count increases
    play_count += 1

    # Prepare game
    black_model.set_team(BLACK)
    white_model.set_team(WHITE)
    
    your_model.set_team(BLACK)
    #your_model.set_team(WHITE)

    if (team == BLACK) :
        game_black_model = your_model
        game_white_model = white_model
    else :
        game_black_model = black_model
        game_white_model = your_model

    # Play game
    winner, board, _ , black_win_prob, white_win_prob, black_go, white_go = game_colosseum.Game_Collosseum(game_black_model, game_white_model, play_count)

    # Record History
    if (winner is BLACK) :
        Record(str("Winner is BLACK - Game : " + str(play_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "pvc_result/game_results.txt")
    else :
        Record(str("Winner is WHITE - Game : " + str(play_count) + " B : " + str(black_go) + " W : " + str(white_go) + "\n"), "pvc_result/game_results.txt")

    Draw_Plot(board, save_image=True, save_image_name=str("pvc_result/snapshot/snapshot_" + str(play_count) + ".png"))
    Draw_Win_Graph(white_win_prob, figure_num=5, save_image_name=str("pvc_result/white_win/white_win_" + str(play_count) + ".png"))
    Draw_Win_Graph(black_win_prob, figure_num=6, save_image_name=str("pvc_result/black_win/black_win_" + str(play_count) + ".png"))

    if (winner is BLACK) :
        black_win += 1
    else :
        white_win += 1
    
    Record(str(" BLACK WIN : "  + str(black_win) + " | WHITE WIN : " + str(white_win) + "\n"),  "pvc_result/win_count.txt")

    del black_win_prob, board, white_win_prob
    