from keras.layers import Conv2D, Dense, Flatten, Input
from keras.models import Model
from keras.models import load_model
from keras import regularizers

import numpy as np
import random as rnd

from mcts import MCTS_Node, Root_Node

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT
from definitions import PARTIAL_OBSERVABILITY, MCTS_ADDITIONAL_SEARCH
from definitions import SAVE_PERIOD
from definitions import WIN_TRAIN_COUNT, LOSE_TRAIN_COUNT

def save_latest_checkpoint_number(filename, number) :
    file = open(filename, "w")
    file.write(str(number))
    file.close()

    return

def get_latest_checkpoint_number(filename) :

    try :
        file = open(filename, "r")
        latest_checkpoint_num = int(file.readline())
        file.close

        return latest_checkpoint_num

    except IOError :
        print("read_checkpoint_fail")
        return -1

class RL_Model():

    # Initialize Function
    def __init__(self, model_name="NULL", print_summary=False) :
        self.model_name = model_name

        self.model, self.losses, self.loss_weights = self._create_model(model_name = model_name)
        self.model.compile(optimizer='adam', loss=self.losses , metrics=['accuracy'] , loss_weights=self.loss_weights)

        if (print_summary is True) :
            self.model.summary()

        self.move_count = 0

    # Create Keras Model (Return Sequential keras layer)
    #  -- Model --
    #  - if Board size is 19
    #  1. Input  : 19 x 19 x PARTIAL_OBSERVABILITY
    #  2. Conv 1 : 19 x 19 x 8  , 8  filters, 3 x 3 , stride 1
    #  3. Conv 2 : 10 x 10 x 16 , 16 filters, 3 x 3 , stride 2
    #  4. Conv 3 : 10 x 10 x 32 , 32 filters, 3 x 3 , stride 1
    #  5. Flatten
    #  6. Dense  : 512, Relu
    #  7. Output : (Spot -  [0 : 19 x 19] - 0 ~ 1) / (WinProb - [0 : 1] - 0 ~ 1
    def _create_model(self, model_name) :
        input_buffer = Input(shape=(BOARD_SIZE, BOARD_SIZE, PARTIAL_OBSERVABILITY))
 
        conv1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='elu')(input_buffer)
        conv2 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), activation='elu')(conv1)
        conv3 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='elu')(conv2)
        
        flatten_layer = Flatten()(conv3)

        last_dense = Dense(512, activation='elu')(flatten_layer)

        spot_prob = Dense(SQUARE_BOARD_SIZE, kernel_regularizer=regularizers.l2(0.001), activation='elu', name="spot_prob")(last_dense)
        win_prob = Dense(1, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid', name="win_prob")(last_dense)

        model = Model(inputs=input_buffer, outputs=[spot_prob, win_prob])

        losses = {"spot_prob" : "categorical_crossentropy",
                  "win_prob"  : "mean_squared_error"}

        loss_weights = {"spot_prob" : 5e-5,
                        "win_prob"  : 1e-5}        

        return model , losses , loss_weights


    ### Set / Get Functions ###
    # Set Team & initialize move count
    # Black = 1, White : -1
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
    # return next_movement's probs & shape
    def predict_func(self, part_obs_inputs_array) :
        part_obs_inputs_array = part_obs_inputs_array.reshape(1, BOARD_SIZE, BOARD_SIZE, PARTIAL_OBSERVABILITY)
        
        return self.model.predict(part_obs_inputs_array)


    ### Train / Test Functions ###
    # Train with results
    # winner is 1 if machine is win, else -1
    def train_func(self, winner) :
        winner = 0.5 + (winner * self.team * 0.5) # Sigmoid : 0 ~ 1
        winner_array = [winner] * self.move_count

        if (len(self.part_obs_inputs_array) != self.move_count) or (len(self.mcts_records) != self.move_count) :
            print("Train func, value error : ", self.move_count, len(self.part_obs_inputs_array), len(self.mcts_records))

        if (int(winner) is 1) :
            try_epochs = WIN_TRAIN_COUNT
        else :
            try_epochs = LOSE_TRAIN_COUNT
            
        winner_array = np.array(winner_array)
        # change (-1, 4, 19, 19)[Channel first] to (-1, 19, 19, 4)[Channel last]
        part_obs_inputs_array = np.transpose(np.array(self.part_obs_inputs_array), (0, 2, 3, 1))
        part_obs_inputs_array = (part_obs_inputs_array * 0.5) + 0.5
        mcts_records = np.array(self.mcts_records).reshape(-1, SQUARE_BOARD_SIZE)

        print(winner_array.shape, part_obs_inputs_array.shape, mcts_records.shape)

        self.model.fit(part_obs_inputs_array, # Input
                       {"spot_prob" : mcts_records, "win_prob"  : winner_array}, # Outputs
                       epochs=try_epochs)
                       
    # Get Current Board's 4 latest states
    # Search Proper movement with MCTS
    # return Next Stone's state
    def get_move(self, my_color, GameState) :
        if (GameState.is_done() is True) :
            return None

        part_obs_inputs = GameState.show_4_latest_boards()

        copy_GameState = GameState.copy()

        root_node = Root_Node(self, copy_GameState, my_color)
        root_node.play_mcts()

        ret_locate = root_node.select_best_child_without_ucb()

        #move, move_prob = root_node.get_spot_with_prob()
        mcts_move_prob = root_node.get_spot_prob().flatten()

        #ret_locate = rnd.choices(population=np.array(move), weights=np.array(move_prob), k=1)[0]
        #ret_locate = tuple(ret_locate)

        root_node.summary()
        win_prob = root_node.get_max_win_prob()

        self.part_obs_inputs_array.append(part_obs_inputs)
        self.mcts_records.append(mcts_move_prob)

        root_node.delete() # copy_GameState will be deleted at here

        if (self.team == BLACK) :
            team_string = "BLACK"
        else :
            team_string = "WHITE"

        print("model team : ", str(team_string), " has move : ", ret_locate)
        self.move_count += 1

        return ret_locate, win_prob

    def save_model(self, num_train_counts) :
        save_checkpoint="result/checkpoint.txt"
        keras_save_name= str("save_" + str(num_train_counts))

        save_latest_checkpoint_number(save_checkpoint, num_train_counts)

        self.model.save(keras_save_name)

        print("save model : ", save_checkpoint, keras_save_name, str(num_train_counts))

        return

    def restore_model(self, restore_train_counts = None) :
        save_checkpoint="result/checkpoint.txt"

        if (restore_train_counts is None) :
            restore_train_counts = get_latest_checkpoint_number(save_checkpoint)

        if (restore_train_counts < 0) :
            return 0

        restore_train_counts = int(restore_train_counts / SAVE_PERIOD) * SAVE_PERIOD

        keras_load_name= str("save_" + str(restore_train_counts))

        self.model = load_model(keras_load_name)

        print("restore model : ", save_checkpoint, keras_load_name, restore_train_counts)
        
        return restore_train_counts

