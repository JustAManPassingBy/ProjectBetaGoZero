import model
import game_colosseum

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import SAVE_PERIOD
from definitions import DEFAULT_PLAY_GAME_COUNTS

from record_functions import Draw_Plot, Draw_Win_Graph, Record
from train_samples import Train_Sample

def main_init_model(TS) :
    # Create Models
    current_model = model.RL_Model(TS)
    current_model.initialize_model()

    # Restore models from "restore_point"
    train_count = current_model.restore_model()

    return train_count, current_model

def load_model(restore_point, TS) :
    # Create Models
    target_model = model.RL_Model(TS)
    target_model.initialize_model()

    # Restore model from restore_point
    train_count = target_model.restore_model(restore_index=restore_point)

    assert(train_count == restore_point)

    return target_model


if __name__ == "__main__" :
    TS = Train_Sample()
    TS.load_all()

    train_count, current_model = main_init_model(TS)
    new_model = load_model(train_count, TS)

    black_win = 0
    white_win = 0

    Record(str("----------------------------- : "  + str(train_count) + "\n"),  "result/log.txt")

    while (True) : 
        train_count += 1

        print("Train : ", train_count)
        Record(str("Train : "  + str(train_count) + "\n"),  "result/log.txt")
        
        new_model.initialize_model()

        # Play game
        winner, board, _ , black_win_prob, white_win_prob, black_go, white_go = game_colosseum.Game_Collosseum(new_model, new_model, train_count)

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
        new_model.train_func(winner)

        # Fight current & new model
        # And save new model if new model is win
        if (train_count % SAVE_PERIOD) is 0 :
            new_model_win, cur_model_win = game_colosseum.Games_Collosseum(new_model, current_model, train_count, DEFAULT_PLAY_GAME_COUNTS / 2)
            tmp_cur_model_win, tmp_new_model_win = game_colosseum.Games_Collosseum(current_model, new_model, train_count, (DEFAULT_PLAY_GAME_COUNTS + 1) / 2)
    
            new_model_win += tmp_new_model_win
            cur_model_win += tmp_cur_model_win

            print(" Train Count " + str(train_count) + " new win : " + str(new_model_win) + " cur win : " + str(cur_model_win))
            Record(str(" Train Count " + str(train_count) + " new win : " + str(new_model_win) + " cur win : " + str(cur_model_win) + "\n"),  "result/log.txt")

            if (new_model_win > cur_model_win) :
                print("save new model")
                Record(str("save new model" + "\n"),  "result/log.txt")

                new_model.save_model(train_count)
                TS.save_all()

                del current_model
                current_model = load_model(train_count, TS)
            else :
                print("abolish new model")
                Record(str("abolish new model" + "\n"),  "result/log.txt")

                train_count -= 10
                TS.save_all()

                del new_model
                new_model = load_model(train_count, TS)

            

