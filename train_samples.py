import numpy as np
import copy

from definitions import MAX_SAMPLE_COUNT
from definitions import MAX_TURN
from definitions import PARTIAL_OBSERVABILITY
from definitions import NUM_SAMPLE_POOLS


# These class stacks train samples for each game, and return stacked train samples.
# You can limit number of sample swith "MAX_SAMPLE_COUNT"

# You can set array pool with "NUM_SAMPLE_POOLS"

class Train_Sample() :
    def __init__(self) :
        self._init()

    def _init(self) :
        self.num_train_item = []
        self.winner_array = []
        self.part_obs_inputs_array = []
        self.mcts_records = []

        for _ in range(0, NUM_SAMPLE_POOLS) : # White Win : 0 / Black Win : 1
            self.winner_array.append([])
            self.num_train_item.append(0)
            self.part_obs_inputs_array.append([])
            self.mcts_records.append([])

    def initialize(self) :
        del self.winner_array
        del self.part_obs_inputs_array
        del self.mcts_records
        self._init()

    def create_8_direction(self, 
                           sample_winner_item,
                           sample_part_obs_inputs_item,
                           sample_mcts_records) :
        # - Rechange (X, Y, PARTIAL_OBSERVABILITY) to (PARTIAL_OBSABILITY, X, Y)
        reshape_sample_part_obs_inputs_item = np.transpose(sample_part_obs_inputs_item, (2, 0, 1))

        # - Create 8 Direction Rotate
        dir_8_sample_winner_item = []
        dir_8_sample_part_obs_inputs_item = []
        dir_8_sample_mcts_records = []

        # - Create 8 Dirction Rotate
        # - use deepcopy 
        for i in range(0, 8) : 
            new_sample_winner_item = copy.deepcopy(sample_winner_item)
            new_sample_part_obs_inputs_item = copy.deepcopy(reshape_sample_part_obs_inputs_item)
            new_sample_mcts_records = copy.deepcopy(sample_mcts_records)

            # - Rotate each partial observability's item
            for j in range(0, PARTIAL_OBSERVABILITY) : 
                if (3 < i) :
                    new_sample_part_obs_inputs_item[j] = np.flip(new_sample_part_obs_inputs_item[j], axis=0)
                
                new_sample_part_obs_inputs_item[j] = np.rot90(new_sample_part_obs_inputs_item[j], i % 4)

            # - Rechange (PARTIAL_OBSABILITY, X, Y) to (X, Y, PARTIAL_OBSABILITY)
            new_sample_part_obs_inputs_item = np.transpose(new_sample_part_obs_inputs_item, (1, 2, 0))

            dir_8_sample_winner_item.append(new_sample_winner_item)
            dir_8_sample_part_obs_inputs_item.append(new_sample_part_obs_inputs_item)
            dir_8_sample_mcts_records.append(new_sample_mcts_records)
        
        return np.asarray(dir_8_sample_winner_item), np.asarray(dir_8_sample_part_obs_inputs_item), np.asarray(dir_8_sample_mcts_records)

                        
    # Stack one sample.
    # You might change this function to get multiple inputs.
    def add(self, 
            sample_winner_array, 
            sample_part_obs_inputs_array, 
            sample_mcts_records,
            sample_index=0) : # Default sample index should be 0

        # Check
        if (sample_index >= MAX_TURN) :
            print(" Error : Sample index !! ", str(sample_index))
            return  
  
        # Expand 8 data
        new_wa, new_poia, new_mr = self.create_8_direction(sample_winner_array, sample_part_obs_inputs_array, sample_mcts_records)

        for i in range(0, len(new_wa)) :
            self.winner_array[sample_index].append(new_wa[i])
            self.part_obs_inputs_array[sample_index].append(new_poia[i])
            self.mcts_records[sample_index].append(new_mr[i])
            self.num_train_item[sample_index] += 1

            if (self.num_train_item[sample_index] > MAX_SAMPLE_COUNT) :
                self.winner_array[sample_index].pop()
                self.part_obs_inputs_array[sample_index].pop()
                self.mcts_records[sample_index].pop()
                self.num_train_item[sample_index] -= 1

        ''' Legacy Codes (Will be deleted)
        self.winner_array[sample_index].append(sample_winner_array)
        self.part_obs_inputs_array[sample_index].append(sample_part_obs_inputs_array)
        self.mcts_records[sample_index].append(sample_mcts_records)
        self.num_train_item[sample_index] += 1

        if (self.num_train_item[sample_index] > MAX_SAMPLE_COUNT) :
            self.winner_array[sample_index].pop()
            self.part_obs_inputs_array[sample_index].pop()
            self.mcts_records[sample_index].pop()
            self.num_train_item[sample_index] -= 1
        '''

        return            

    # Decay current sample's winning point
    def decay(self, 
              start_idx = None,
              end_idx = None,
              decay_ratio = 0.999,
              sample_index=0) : # Default sample index should be 0
        if (start_idx is None) :
            start_idx = 0

        if (end_idx is None) :
            end_idx = len(self.winner_array[sample_index])

        for i in range(start_idx, end_idx) :
            self.winner_array[sample_index][i] = 0.5 + (self.winner_array[sample_index][i] - 0.5) * decay_ratio

        return

    # return stacked train samples.
    def get(self,
            sample_index=0) : # Default sample index should be 0
        return np.asarray(self.winner_array[sample_index]), np.asarray(self.part_obs_inputs_array[sample_index]) , np.asarray(self.mcts_records[sample_index])

    def get_all(self) :
        all_winner_array = []
        all_part_obs_inputs_array = []
        all_mcts_records = []

        for idx in range(0, NUM_SAMPLE_POOLS) :
            all_winner_array.extend(self.winner_array[idx])
            all_part_obs_inputs_array.extend(self.part_obs_inputs_array[idx])
            all_mcts_records.extend(self.mcts_records[idx])

        return np.asarray(all_winner_array), np.asarray(all_part_obs_inputs_array) , np.asarray(all_mcts_records)