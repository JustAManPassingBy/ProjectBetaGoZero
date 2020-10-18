from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from keras.models import load_model
from keras import regularizers

import numpy as np
import random as rnd

from mcts import MCTS_Node, Root_Node

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE
from definitions import PARTIAL_OBSERVABILITY, ADDITIONAL_SEARCH_COUNT_DURING_MCTS
from definitions import SAVE_PERIOD
from definitions import WIN_TRAIN_COUNT, LOSE_TRAIN_COUNT
from definitions import MAX_TURN
from definitions import RANDOM_MOVE_PROB
from definitions import MCTS_N_ITERS

from train_samples import Train_Sample
import time

from game_hub import *
from layer import *

# Save latest checkpoint number (for restores)
def save_latest_checkpoint_number(filename, number) :
    file = open(filename, "w")
    file.write(str(number))
    file.close()

    return

# Restore latest checkpoint number
def get_latest_checkpoint_number(filename) :

    try :
        file = open(filename, "r")
        latest_checkpoint_num = int(file.readline())
        file.close

        return latest_checkpoint_num

    except IOError :
        print("read_checkpoint_fail")
        return -1


### Model's Class
class RL_Model():

    # Initialize Function
    def __init__(self, model_name="NULL", print_summary=False) :
        self.model_name = model_name

        # Create models for each move
        model, losses, loss_weights = self._create_model(model_name = model_name)
        model.compile(optimizer='adam', loss=losses , metrics=['accuracy'] , loss_weights=loss_weights)

        if (print_summary is True) :
            model.summary()

        self.model = model

        self.move_count = 0
        self.Train_Sample = Train_Sample()

    # Create Keras Model (Return Sequential keras layer)
    def _create_model(self, model_name) :
        # Defined at layer.py
        return create_keras_layer()


    ### Set / Get Functions ###
    # Set Team & initialize move count
    # Model user should call this function to initialize previous game states. (move count, stones, ...)
    def set_team(self, team) :
        self.team = team
        self.move_count = 0

        self.part_obs_inputs_array = [] # Records of inputs consider machine's team
        self.mcts_records = []   # Records of MCTS Result (for spot_prob)

    # Get Team
    def get_team(self) :
        return self.team

    # Get Move Count
    def get_move_count(self) :
        return self.move_count

    # Get Current Board's (PARTIAL_OBSERVABILITY) latest states
    def predict_func(self, part_obs_inputs_array, move_count) :
        part_obs_inputs_array = part_obs_inputs_array.reshape(1, BOARD_SIZE, BOARD_SIZE, PARTIAL_OBSERVABILITY)
        
        if (move_count > MAX_TURN) :
            return

        # return next_movement's probs & shape
        return self.model.predict(part_obs_inputs_array)


    # Record opposite model
    # - This model will be used at mcts -> expand
    def hang_opposite_model(self, opposite_Model) :
        self.opposite_Model = opposite_Model

        return


    ### Train / Test Functions ###
    # Train with results
    def train_func(self, winner) :
        print("Model : ", str(self.team), " winner val : ", str(winner))

        # Black win : 1 -> 1
        # White Win : -1 -> 0
        # Same win prob = 0 -> 0.5
        winner = 0.5 + (winner * 0.5) # Sigmoid : [-1 ~ 1] -> [0 ~ 1]
        
        # - Increasing win scores.
        #  At the first of the game, win probability is not precise.
        #  And when game reaches to end, win probability is obvious.
        #  So decreasing "win probability" at the beginning of the game, and maximize it at end of the game.
        winner_start = 0.2 + 0.6 * winner # 0.6 for win , 0.4 for lose
        #winner_start = winner
        winner_end = winner
        winner_div = (winner_end - winner_start) / float(self.move_count)

        winner_array = []

        for i in range(0, len(self.part_obs_inputs_array)) :
            result = winner_start + winner_div * i
            winner_array.append(result)

        # - Error check
        if (len(self.part_obs_inputs_array) != len(self.mcts_records)) :
            print("Train func, value error : ", self.move_count, len(self.part_obs_inputs_array), len(self.mcts_records))
            return

        # - Option : you can set winner's and loser's train count(epochs) different.
        if (int(winner) is 1) :
            try_epochs = WIN_TRAIN_COUNT
        else :
            try_epochs = LOSE_TRAIN_COUNT
            
        # - Change (-1, 4, X, X)[Channel first] to (-1, X, X, 4)[Channel last]
        part_obs_inputs_array = np.transpose(np.array(self.part_obs_inputs_array), (0, 2, 3, 1))
        mcts_records = np.array(self.mcts_records).reshape(-1, SQUARE_BOARD_SIZE + 1)
        winner_array = np.array(winner_array)

        print("Train Once")

        self.model.fit(part_obs_inputs_array, # Input
                    {"spot_prob" : mcts_records, "win_prob"  : winner_array}, # Outputs
                    epochs=try_epochs)

        print("Add New samples into sample pool")
        #self.Train_Sample.decay(sample_index=WHITE)
        #self.Train_Sample.decay(sample_index=BLACK)

        for idx in range(0, self.move_count) :
            if (idx >= MAX_TURN) :
                break

            # Seperate black win's pool and white win's pool
            self.Train_Sample.add(winner_array[idx], part_obs_inputs_array[idx], mcts_records[idx], int(winner))

        t_winner_array, t_part_obs_inputs_array, t_mcts_records = self.Train_Sample.get_all()

        print("Train")
        self.model.fit(t_part_obs_inputs_array, # Input
                       {"spot_prob" : t_mcts_records, "win_prob"  : t_winner_array}, # Outputs
                       epochs=try_epochs)
                       
    # Get Current Board's N(PARTIAL_OBSERVABILITY) latest states
    # Search Proper movement with MCTS(mcts.py)
    # return Next Stone's state and win probability
    def get_move(self, my_color, GameState, debug_mode=0) :
        # - Check whether use debug mode
        if (debug_mode is not 0) :
            return self.get_move_debug(my_color, GameState, debug_mode)

        # - Check return condition : Obviously game is end.
        if (GameState.is_done() is True) :
            return None, 0.0, []

        # - Get Board's latest #'s shape (Input)
        part_obs_inputs = get_game_state(GameState)
        #part_obs_inputs = GameState.show_4_latest_boards()

        # - Copy Gamestate
        copy_GameState = GameState.copy()

        # - MCTS Search(play)
        root_node = Root_Node(self, copy_GameState, my_color, self.move_count)
        root_node.play_mcts()

        # - Get best move(action / child)
        ret_locate = root_node.select_best_child_without_ucb(my_color).action
        ret_actions = root_node._calculate_best_actions(my_color)

        print(ret_actions)

        # - Record current move's database
        mcts_move_prob = root_node.get_spot_prob().flatten()

        root_node.summary()

        win_prob = root_node.get_max_win_prob()
        self.part_obs_inputs_array.append(part_obs_inputs)
        self.mcts_records.append(mcts_move_prob)

        self.opposite_Model.part_obs_inputs_array.append(part_obs_inputs)
        self.opposite_Model.mcts_records.append(mcts_move_prob)

        # - Delete original game state
        root_node.delete()

        # - Print game information
        if (self.team == BLACK) :
            team_string = "BLACK"
        else :
            team_string = "WHITE"
        print("model team : ", str(team_string), " has move : ", ret_locate, " count : ", self.move_count)

        self.move_count += 1

        return ret_locate, win_prob, ret_actions

    # return debug function specified in "debug_mode"
    # If debug_mode is 0, return NULL, NULL
    def get_move_debug(self, my_color, GameState, debug_mode) :
        # If debug mode
        if (debug_mode == 1) :

             # - Check return condition : Obviously game is end.
            if (GameState.is_done() is True) or (self.move_count >= (MAX_TURN - ADDITIONAL_SEARCH_COUNT_DURING_MCTS)) :
                return None, None

            part_obs_inputs = get_game_state(GameState)
            #part_obs_inputs = GameState.show_4_latest_boards()

            # - Create MCTS & MCTS play
            copy_GameState = GameState.copy()

            root_node = Root_Node(self, copy_GameState, my_color, self.move_count)

            print("-- initial --")
            print(np.around(root_node.get_win_prob(), decimals=3))
            print(" -----------")
            print(root_node.get_spot_count())

            print("-- After train 300 -- ")
            root_node.play_mcts(n_iters=300)
            print(np.around(root_node.get_win_prob(), decimals=3))
            print(" -----------")
            print(root_node.get_spot_count())

            print("-- After train 700 -- ")
            root_node.play_mcts(n_iters=700)
            print(np.around(root_node.get_win_prob(), decimals=3))
            print(" -----------")
            print(root_node.get_spot_count())

            print("-- After train 1000 -- ")
            root_node.play_mcts(n_iters=1000)
            print(np.around(root_node.get_win_prob(), decimals=3))
            print(" -----------")
            print(root_node.get_spot_count())

            print("-- After train 3000 -- ")
            root_node.play_mcts(n_iters=3000)
            print(np.around(root_node.get_win_prob(), decimals=3))
            print(" -----------")
            print(root_node.get_spot_count())

            # - From results of MCTS Search, get best move or random move
            ret_locate = root_node.select_best_child_without_ucb(my_color).action
            ret_actions = root_node._calculate_best_actions(my_color)

            mcts_move_prob = root_node.get_spot_prob().flatten()

            root_node.summary()

            win_prob = root_node.get_max_win_prob()
            self.part_obs_inputs_array.append(part_obs_inputs)
            self.mcts_records.append(mcts_move_prob)

            # - Delete original game state
            root_node.delete()

            # - Print game information
            if (self.team == BLACK) :
                team_string = "BLACK"
            else :
                team_string = "WHITE"
            print("model team : ", str(team_string), " has move : ", ret_locate, " count : ", self.move_count)

            self.move_count += 1

            return ret_locate, win_prob, ret_actions

        elif (debug_mode == 2) :
            # - Check return condition : Obviously game is end.
            if (GameState.is_done() is True) :
                return None, 0.0, []

            # - Get Board's latest #'s shape (Input)
            part_obs_inputs = get_game_state(GameState)
            #part_obs_inputs = GameState.show_4_latest_boards()

            # - Copy Gamestate
            copy_GameState = GameState.copy()

            # - MCTS Search(play)
            root_node = Root_Node(self, copy_GameState, my_color, self.move_count, debug_mode=debug_mode)

            print(" -- Input -- ")
            print(part_obs_inputs)

            iter_times = 1

            while (iter_times > 0) :
                iter_times = int(input(" Run count : "))

                if (iter_times > 0) :
                    root_node.play_mcts(n_iters=iter_times)

            # - Get best move(action / child)
            ret_locate = root_node.select_best_child_without_ucb(my_color).action
            ret_actions = root_node._calculate_best_actions(my_color)

            print(ret_actions)

            # Check whether return is none
            if (ret_locate is None) :
                return None, 0.0, []

            # - Record current move's database
            mcts_move_prob = root_node.get_spot_prob().flatten()

            root_node.summary()

            win_prob = root_node.get_max_win_prob()
            self.part_obs_inputs_array.append(part_obs_inputs)
            self.mcts_records.append(mcts_move_prob)

            self.opposite_Model.part_obs_inputs_array.append(part_obs_inputs)
            self.opposite_Model.mcts_records.append(mcts_move_prob)

            # - Delete original game state
            root_node.delete()

            # - Print game information
            if (self.team == BLACK) :
                team_string = "BLACK"
            else :
                team_string = "WHITE"
            print("model team : ", str(team_string), " has move : ", ret_locate, " count : ", self.move_count)

            self.move_count += 1

            return ret_locate, win_prob, ret_actions

        else :
            print("[Error] Wrong debug mode : ", debug_mode)
            return None, None

    ### SAVE & Restore Models
    def save_model(self, 
                   num_train_counts) :
        save_checkpoint="result/checkpoint.txt"
        save_latest_checkpoint_number(save_checkpoint, num_train_counts)

        keras_save_name = str("result/save_" + str(num_train_counts) + "_" + str(self.team))
        self.model.save(keras_save_name)

        print("save model : ", save_checkpoint, keras_save_name, str(num_train_counts))

        return

    def restore_model(self, 
                      restore_point=None) :
        
        save_checkpoint="result/checkpoint.txt"

        if (restore_point is None) :
            restore_point = get_latest_checkpoint_number(save_checkpoint)

        if (restore_point < 0) :
            return 0

        restore_point = int(restore_point / SAVE_PERIOD) * SAVE_PERIOD

        keras_load_name= str("result/save_" + str(restore_point) + "_" + str(self.team))
        self.model = load_model(keras_load_name)

        print("restore model : ", save_checkpoint, keras_load_name, restore_point)
        
        return restore_point

