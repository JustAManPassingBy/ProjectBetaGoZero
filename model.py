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

from train_samples import Train_Sample

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
    #  -- Model --
    #  - if Board size is 19
    #  1. Input   : 19 x 19 x PARTIAL_OBSERVABILITY
    #  2. Conv 1  : 19 x 19 x 8  , 8  filters, 3 x 3 , stride (1, 1)
    #  3. Conv 2  : 10 x 19 x 16 , 16 filters, 3 x 3 , stride (2, 1)
    #  4. Conv 3  : 10 x 19 x 32 , 32 filters, 3 x 3 , stride (1, 1)
    #  5. Conv 4  : 10 x 10 x 64 , 64 filters, 3 x 3 , stride (1, 2)
    #  5. Flatten
    #  6. Dense 1 : 4096, Relu
    #  6. Dense 2 : 1024, Relu
    #  7. Output : (Spot -  [0 : 19 x 19] - 0 ~ 1) / (WinProb - [-1 : 1] - -1 ~ 1
    def _create_model(self, model_name) :
        input_buffer = Input(shape=(BOARD_SIZE, BOARD_SIZE, PARTIAL_OBSERVABILITY))
 
        conv1 = Conv2D(8, (3, 3), padding='same', strides=(1, 1), activation='elu')(input_buffer)
        conv2 = Conv2D(16, (3, 3), padding='same', strides=(2, 1), activation='elu')(conv1)
        conv3 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='elu')(conv2)
        conv4 = Conv2D(64, (3, 3), padding='same', strides=(1, 2), activation='elu')(conv3)
        
        flatten_layer = Flatten()(conv4)

        dense1 = Dense(4096, activation='elu')(flatten_layer)
        last_dense = Dense(1024, activation='elu')(dense1)

        spot_prob = Dense(SQUARE_BOARD_SIZE, kernel_regularizer=regularizers.l2(1e-2), activation='sigmoid', name="spot_prob")(last_dense)
        win_prob = Dense(1, kernel_regularizer=regularizers.l2(1e-3), activation='sigmoid', name="win_prob")(last_dense)

        model = Model(inputs=input_buffer, outputs=[spot_prob, win_prob])

        losses = {"spot_prob" : "categorical_crossentropy",
                  "win_prob"  : "mean_squared_error"}

        loss_weights = {"spot_prob" : 1e-3,
                        "win_prob"  : 1e-5}        

        return model , losses , loss_weights


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

        winner = winner * self.team

        winner = 0.5 + (winner * 0.5) # Sigmoid : [-1 ~ 1] -> [0 ~ 1]
        
        # - Increasing win scores.
        #  At the first of the game, win probability is not precise.
        #  And when game reaches to end, win probability is obvious.
        #  So decreasing "win probability" at the beginning of the game, and maximize it at end of the game.
        # (No use) winner_start = 0.4 + 0.2 * winner # 0.6 for win , 0.4 for lose
        winner_start = winner
        winner_end = winner
        winner_div = (winner_end - winner_start) / float(self.move_count)

        winner_array = []

        for i in range(0, self.move_count) :
            result = winner_start + winner_div * i
            winner_array.append(result)

        if (len(self.part_obs_inputs_array) != self.move_count) or (len(self.mcts_records) != self.move_count) :
            print("Train func, value error : ", self.move_count, len(self.part_obs_inputs_array), len(self.mcts_records))
            return

        # - Option : you can set winner's and loser's train count(epochs) different.
        if (int(winner) is 1) :
            try_epochs = WIN_TRAIN_COUNT
        else :
            try_epochs = LOSE_TRAIN_COUNT
            
        winner_array = np.array(winner_array)
        # - Change (-1, 4, X, X)[Channel first] to (-1, X, X, 4)[Channel last]
        part_obs_inputs_array = np.transpose(np.array(self.part_obs_inputs_array), (0, 2, 3, 1))

        mcts_records = np.array(self.mcts_records).reshape(-1, SQUARE_BOARD_SIZE)

        print("Train Once")

        self.model.fit(part_obs_inputs_array, # Input
                    {"spot_prob" : mcts_records, "win_prob"  : winner_array}, # Outputs
                    epochs=try_epochs)

        print("Add New samples into sample pool")
        self.Train_Sample.decay()

        for idx in range(0, self.move_count) :
            if (idx >= MAX_TURN) :
                break

            self.Train_Sample.add(winner_array[idx], part_obs_inputs_array[idx], mcts_records[idx])

        t_winner_array, t_part_obs_inputs_array, t_mcts_records = self.Train_Sample.get()

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
        if (GameState.is_done() is True) or (self.move_count >= (MAX_TURN - ADDITIONAL_SEARCH_COUNT_DURING_MCTS)) :
            return None, None

        part_obs_inputs = GameState.show_4_latest_boards()

        # - Create MCTS & MCTS play
        copy_GameState = GameState.copy()

        root_node = Root_Node(self, copy_GameState, my_color, self.move_count)
        root_node.play_mcts()

        # - From results of MCTS Search, get best move
        ret_locate = root_node.select_best_child_without_ucb()

        # !! Check ret_locate is none 
        # !! It's better why this condition was occured.
        if (ret_locate is None) :
            return None, None

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

        return ret_locate, win_prob

    # return debug function specified in "debug_mode"
    # If debug_mode is 0, return get_move()
    def get_move_debug(self, my_color, GameState, debug_mode) :
        # If debug mode
        if (debug_mode == 1) :

             # - Check return condition : Obviously game is end.
            if (GameState.is_done() is True) or (self.move_count >= (MAX_TURN - ADDITIONAL_SEARCH_COUNT_DURING_MCTS)) :
                return None, None

            part_obs_inputs = GameState.show_4_latest_boards()

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


            # - From results of MCTS Search, get best move
            ret_locate = root_node.select_best_child_without_ucb()

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

            return ret_locate, win_prob

        else :
            print("[Error] why debug_mode is 0 at debug function ??")
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

