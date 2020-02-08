import numpy as np

from definitions import MAX_SAMPLE_COUNT



class Train_Sample() :
    def __init__(self) :
        self._init()

    def _init(self) :
        self.num_train_item = 0
        self.winner_array = []
        self.part_obs_inputs_array = []
        self.mcts_records = []

    def initialize(self) :
        del self.winner_array
        del self.part_obs_inputs_array
        del self.mcts_records
        self._init()


    def add(self, 
            sample_winner_array, 
            sample_part_obs_inputs_array, 
            sample_mcts_records) :
        # Check
        if (len(sample_mcts_records) != len(sample_part_obs_inputs_array)) or (len(sample_mcts_records) != len(sample_winner_array)) :
            print(" Error - Sample : Length is different" , len(sample_mcts_records), len(sample_part_obs_inputs_array), len(sample_winner_array))
            return      

        for i in range(0, len(sample_mcts_records)) :
            self.winner_array.append(sample_winner_array[i])
            self.part_obs_inputs_array.append(sample_part_obs_inputs_array[i])
            self.mcts_records.append(sample_mcts_records[i])
            self.num_train_item += 1

            if (self.num_train_item > MAX_SAMPLE_COUNT) :
                self.winner_array.pop(0)
                self.part_obs_inputs_array.pop()
                self.mcts_records.pop(0)
                self.num_train_item -= 1

        return            


    def get(self) :
        return np.asarray(self.winner_array), np.asarray(self.part_obs_inputs_array) , np.asarray(self.mcts_records)