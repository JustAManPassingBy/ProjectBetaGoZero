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

        self.model = []

        # Create models for each move
        # - Each move's probability will be calculated with different models.
        for _ in range(0, MAX_TURN) :
            model, losses, loss_weights = self._create_model(model_name = model_name)
            model.compile(optimizer='adam', loss=losses , metrics=['accuracy'] , loss_weights=loss_weights)
            self.model.append(model)

        if (print_summary is True) :
            model.summary()

        self.move_count = 0
        self.Train_Sample = Train_Sample()

    # Create Keras Model (Return Sequential keras layer)
    #  -- Model --
    #  - if Board size is 19
    #  1. Input  : 19 x 19 x PARTIAL_OBSERVABILITY
    #  2. Conv 1 : 19 x 19 x 8  , 8  filters, 3 x 3 , stride 1
    #  3. Conv 2 : 10 x 10 x 16 , 16 filters, 3 x 3 , stride 2
    #  4. Conv 3 : 10 x 10 x 32 , 32 filters, 3 x 3 , stride 1
    #  5. Flatten
    #  6. Dense  : 512, Relu
    #  7. Output : (Spot -  [0 : 19 x 19] - 0 ~ 1) / (WinProb - [-1 : 1] - -1 ~ 1
    def _create_model(self, model_name) :
        input_buffer = Input(shape=(BOARD_SIZE, BOARD_SIZE, PARTIAL_OBSERVABILITY))
 
        conv1 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='elu')(input_buffer)
        conv2 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), activation='elu')(conv1)
        conv3 = Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='elu')(conv2)
        
        flatten_layer = Flatten()(conv3)

        last_dense = Dense(512, activation='elu')(flatten_layer)

        spot_prob = Dense(SQUARE_BOARD_SIZE, kernel_regularizer=regularizers.l2(1e-3), activation='elu', name="spot_prob")(last_dense)
        win_prob = Dense(1, kernel_regularizer=regularizers.l2(1e-5), activation='elu', name="win_prob")(last_dense)

        model = Model(inputs=input_buffer, outputs=[spot_prob, win_prob])

        losses = {"spot_prob" : "categorical_crossentropy",
                  "win_prob"  : "mean_squared_error"}

        loss_weights = {"spot_prob" : 5e-2,
                        "win_prob"  : 1e-6}        

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
        return self.model[move_count].predict(part_obs_inputs_array)


    # Record opposite model
    # - This model will be used at mcts -> expand
    def hang_opposite_model(self, opposite_Model) :
        self.opposite_Model = opposite_Model

        return


    ### Train / Test Functions ###
    # Train with results
    def train_func(self, winner) :
        print("Model : ", str(self.team), " winner val : ", str(winner))

        ## winner = 0.5 + (winner * self.team * 0.5) # Sigmoid : 0 ~ 1
        
        # - Increasing win scores.
        #  At the first of the game, win probability is not precise.
        #  And when game reaches to end, win probability is obvious.
        #  So decreasing "win probability" at the beginning of the game, and maximize it at end of the game.
        winner_start = 0.2 * 0.4 * winner # 0.2 for win , -0.2 for lose
        winner_end = winner
        winner_div = (winner_end - winner_start) / float(self.move_count)

        winner_array = []

        for i in range(0, self.move_count) :
            result = winner_start + winner_div * i
            winner_array.append(result)

        # - Error check
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

        # (Not use) Change Board Map : -1(black) 0(none) 1(white) -> 0(black) 0.5(none) 1(white)
        #part_obs_inputs_array = (part_obs_inputs_array * 0.5) + 0.5

        mcts_records = np.array(self.mcts_records).reshape(-1, SQUARE_BOARD_SIZE)

        # - Train(fit) each model with board and win probability informations
        for idx in range(0, self.move_count) :
            # - Range of model : 0 ~ (MAX_TURN - 1)
            if (idx >= MAX_TURN) :
                break

            print("Train : ", idx)

            self.Train_Sample.add(winner_array[idx], part_obs_inputs_array[idx], mcts_records[idx], idx)
            t_winner_array, t_part_obs_inputs_array, t_mcts_records = self.Train_Sample.get(idx)

            self.model[idx].fit(t_part_obs_inputs_array, # Input
                                {"spot_prob" : t_mcts_records, "win_prob"  : t_winner_array}, # Outputs
                                epochs=try_epochs)
                       
    # Get Current Board's N(PARTIAL_OBSERVABILITY) latest states
    # Search Proper movement with MCTS(mcts.py)
    # return Next Stone's state and win probability
    def get_move(self, my_color, GameState) :
        # - Check return condition : Obviously game is end.
        if (GameState.is_done() is True) or (self.move_count >= (MAX_TURN - MCTS_ADDITIONAL_SEARCH)) :
            return None, None

        # - Get Current Board's Input
        part_obs_inputs = GameState.show_4_latest_boards()

        # - Copy Gamestate.
        # - This will be used in MCTS search
        copy_GameState = GameState.copy()

        # - Create MCTS & MCTS play
        root_node = Root_Node(self, copy_GameState, my_color, self.move_count)
        root_node.play_mcts()

        # - From results of MCTS Search, get best move
        ret_locate = root_node.select_best_child_without_ucb()

        # !! Check ret_locate is none 
        # !! It's better why this condition was occured.
        if (ret_locate is None) :
            return None, None

        # - Get each move's spot probability 
        #  This data will be used in model's training 
        mcts_move_prob = root_node.get_spot_prob().flatten()

        # (X) Decive next move with random function (no use)
        #ret_locate = rnd.choices(population=np.array(move), weights=np.array(move_prob), k=1)[0]
        #ret_locate = tuple(ret_locate)

        # - Print Summary, stack data.
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


    ### SAVE & Restore Models
    def save_model(self, num_train_counts) :
        save_checkpoint="result/checkpoint.txt"
        save_latest_checkpoint_number(save_checkpoint, num_train_counts)

        for i in range(0, MAX_TURN) :
            keras_save_name = str("result/save_" + str(num_train_counts) + "_" + str(self.team) + "_" + str(i))
            self.model[i].save(keras_save_name)

        print("save model : ", save_checkpoint, keras_save_name, str(num_train_counts))

        return

    def restore_model(self, restore_train_counts = None) :
        save_checkpoint="result/checkpoint.txt"

        if (restore_train_counts is None) :
            restore_train_counts = get_latest_checkpoint_number(save_checkpoint)

        if (restore_train_counts < 0) :
            return 0

        restore_train_counts = int(restore_train_counts / SAVE_PERIOD) * SAVE_PERIOD

        for i in range(0, MAX_TURN) :
            print("restore model ", str(i))
            keras_load_name= str("result/save_" + str(restore_train_counts) + "_" + str(self.team) + "_" + str(i))
            self.model[i] = load_model(keras_load_name)

        print("restore model : ", save_checkpoint, keras_load_name, restore_train_counts)
        
        return restore_train_counts

