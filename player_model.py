import numpy as np
import random as rnd

from definitions import BLACK, WHITE


### Model's Class
class PLAYER_Model():

    # Initialize Function
    def __init__(self, model_name="NULL") :
        self.model_name = model_name

    ### Set / Get Functions ###
    # Set Team & initialize move count
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

    # Do nothing
    def predict_func(self, part_obs_inputs_array, move_count) :
        return

    ### Train / Test Functions ###
    # Do nothing
    def train_func(self, winner) :
        return
                       
    # Get Current Board's N(PARTIAL_OBSERVABILITY) latest states
    # Search Proper movement with MCTS(mcts.py)
    # return Next Stone's state and win probability
    def get_move(self, my_color, GameState) :
        input_array = input("Get None or x , y : ").split()
        
        if (len(input_array) is 1) :
            return None, 1.0

        else :
            ret_val = (int(input_array[0]), int(input_array[-1]))
            print(ret_val)
            return ret_val, 1.0



