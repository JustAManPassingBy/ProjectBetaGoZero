import numpy as np

from definitions import MAX_SAMPLE_COUNT
from definitions import MAX_TURN


# These class stacks train samples for each game, and return stacked train samples.
# You can limit number of sample swith "MAX_SAMPLE_COUNT"

class Train_Sample() :
    def __init__(self) :
        self._init()

    def _init(self) :
        self.num_train_item = []
        self.winner_array = []
        self.part_obs_inputs_array = []
        self.mcts_records = []

        for _ in range(0, MAX_TURN) :
            self.winner_array.append([])
            self.num_train_item.append(0)
            self.part_obs_inputs_array.append([])
            self.mcts_records.append([])

    def initialize(self) :
        del self.winner_array
        del self.part_obs_inputs_array
        del self.mcts_records
        self._init()


    # Stack one sample.
    # You might change this function to get multiple inputs.
    def add(self, 
            sample_winner_array, 
            sample_part_obs_inputs_array, 
            sample_mcts_records,
            sample_index) :
        # Check
        if (sample_index >= MAX_TURN) :
            print(" Error : Sample index !! ", str(sample_index))
            return   

        self.winner_array[sample_index].append(sample_winner_array)
        self.part_obs_inputs_array[sample_index].append(sample_part_obs_inputs_array)
        self.mcts_records[sample_index].append(sample_mcts_records)
        self.num_train_item[sample_index] += 1

        if (self.num_train_item[sample_index] > MAX_SAMPLE_COUNT) :
            self.winner_array[sample_index].pop(0)
            self.part_obs_inputs_array[sample_index].pop(0)
            self.mcts_records[sample_index].pop(0)
            self.num_train_item[sample_index] -= 1

        return            

    # return stacked train samples.
    def get(self,
            sample_index) :
        return np.asarray(self.winner_array[sample_index]), np.asarray(self.part_obs_inputs_array[sample_index]) , np.asarray(self.mcts_records[sample_index])