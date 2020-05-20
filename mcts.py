import numpy as np
import math
import random as rnd

import game

from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, NUM_STONES_TO_END_GAME
from definitions import PARTIAL_OBSERVABILITY, ADDITIONAL_SEARCH_COUNT_DURING_MCTS
from definitions import MAX_TURN
from definitions import MCTS_N_ITERS
from definitions import RANDOM_MOVE_PROB

from record_functions import Draw_Win_Spot

from game_hub import *

# Each node means specific move
#            (Root Node)
#           Initial state
#             /       \
#         (Node 1)   (Node 2)
#          Move 1     Move 2
#          /  \
#           ... 

class MCTS_Node():
    
    def __init__(self, action, parent, prior_prob, turn, move_count, Model, GameState) :
        self.parent = parent # None if Root node
        self.action = action # Previous action (X, Y)
        self.turn = turn     # BLACK : 1 , WHITE = -1
        self.move_count = move_count
        self.Model = Model

        self.GameState = GameState

        self.value_sum = 0.0
        self.times_visited = 0

        self.children = []

        self.possible_actions_calculated = False
        self.action_spot_prob_calculated = False

        self.black_best_action = None
        self.white_best_action = None

        self.prior_prob = prior_prob


    def _is_leaf(self) :
        return len(self.children) is 0


    def _is_root(self) :
        return self.parent is None


    def _get_mean_value(self, turn) :
        self._calculate_action_spot_prob()

        return self.value_sum / self.times_visited # Use Average


    def _get_high_value(self, turn) :
        self._calculate_action_spot_prob()
        
        if (turn is BLACK) :
            val = self.value_max
        else :
            val = self.value_min

        return val


    # UCB Score
    def _ucb_score(self, turn, scale=0.1) :        
        # U(s, a) = scale * P(s, a) / (1 + N(s, a))
        ucb = self.prior_prob / (1 + self.times_visited)
        
        # White's ucb = ucb * -1 (turn)
        # Black's ucb = ucb * 1  (turn)
        scaled_ucb = ucb * scale * turn

        return self._get_mean_value(turn) + scaled_ucb


    def _calculate_possible_actions(self) :
        if (self.possible_actions_calculated is False) :
            self.possible_actions = get_valid_states(self.GameState)
            
            # We have to consider "None" move (It means "do nothing")
            #self.possible_actions.append(None)

            self._calculate_action_spot_prob()

            self.high_prob_actions = self._get_high_prob_actions()

            self.possible_actions_calculated = True


    def _calculate_action_spot_prob(self,
                                    opposite=True) :
        if (self.action_spot_prob_calculated is False) :
            # Check whether use opposite model's value
            if (opposite is True) and (self.turn != self.Model.get_team()) :
                use_opposite_action = True
            else :
                use_opposite_action = False

            if (self.Model.get_team is WHITE) :
                use_opposite_value = True
            else :
                use_opposite_value = False

            # From above values, calculate Our & Opposite model's value, and save it.
            if (use_opposite_action is True) and (use_opposite_value is True) :
                self.action_prob , self.node_init_value = self.Model.opposite_Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count)
            
            elif (use_opposite_action is True) and (use_opposite_value is False) :
                self.action_prob, _ = self.Model.opposite_Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count)
                _, self.node_init_value = self.Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) 
            
            elif (use_opposite_action is False) and (use_opposite_value is True) :
                _ , self.node_init_value = self.Model.opposite_Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count)
                self.action_prob , _  = self.Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) 

            else :
                self.action_prob , self.node_init_value = self.Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) # spot_prob / win_prob

            self.node_init_value = np.asscalar(self.node_init_value)
            self.action_prob = self.action_prob.flatten()

            self.value_sum = self.node_init_value
            self.times_visited += 1

            self.value_min = self.node_init_value
            self.value_max = self.node_init_value

            self.black_best_actions = [self.action]
            self.white_best_actions = [self.action]

            self.action_spot_prob_calculated = True


    def _calculate_spot_prob_from_action(self, action) :
        if (action is None) :
            return 0.0

        (action_x, action_y) = action
        action_idx = action_x + action_y * BOARD_SIZE

        self._calculate_action_spot_prob()

        return self.action_prob[action_idx]


    ##  Functions to get high probability actions
    def _get_high_prob_actions(self, num_actions=12) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        highest_list = np.array(self.action_prob)
        highest_sort = highest_list.argsort()

        for i in range(len(highest_sort)) :
            if (num_actions == 0) :
                break

            highest_action = highest_sort[-(i + 1)]

            # None : None action
            if (highest_action == SQUARE_BOARD_SIZE) :
                highest_move = None
            # Else : action
            else :
                highest_y = int(highest_action / BOARD_SIZE)
                highest_x = int(highest_action % BOARD_SIZE)

                # !! Move for action is (X, Y), not (Y, X)
                highest_move = (highest_x, highest_y)

            if (highest_move in self.possible_actions) or (highest_move is None) :
                high_prob_actions.append(highest_move)
                num_actions -= 1

        return high_prob_actions


    def _get_high_prob_actions_exceed_threshold(self, threshold=float(1 / SQUARE_BOARD_SIZE + 1)) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        highest_list = np.array(self.action_prob)
        highest_sort = highest_list.argsort()

        for i in range(len(highest_sort)) :
            if (highest_list[highest_sort[-(i + 1)]] < threshold) :
                break

            highest = highest_list[highest_sort[-(i + 1)]]
            highest_locate = highest_sort[-(i + 1)]

            # None : None action
            if (highest == SQUARE_BOARD_SIZE) :
                highest_move = None
            # Else : action
            else :
                highest_y = int(highest_locate / BOARD_SIZE)
                highest_x = int(highest_locate % BOARD_SIZE)

                # !! Move for action is (X, Y), not (Y, X)
                highest_move = (highest_x, highest_y)

            if (highest_move in self.possible_actions) or (highest_move is None) :
                high_prob_actions.append(highest_move)
            
        return high_prob_actions


    def _get_high_prob_actions_range_from_max(self, limit=0.1) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        highest_list = np.array(self.action_prob)
        highest_sort = highest_list.argsort()

        for i in range(len(highest_sort)) :
            highest = highest_list[highest_sort[-(i + 1)]]
            highest_locate = highest_sort[-(i + 1)]

            if (i == 0) :
                action_limit = highest - limit

            if (highest < action_limit) :
                break

            # None : None action
            if (highest == SQUARE_BOARD_SIZE) :
                highest_move = None
            # Else : action
            else :
                highest_y = int(highest_locate / BOARD_SIZE)
                highest_x = int(highest_locate % BOARD_SIZE)

                # !! Move for action is (X, Y), not (Y, X)
                highest_move = (highest_x, highest_y)

            if (highest_move in self.possible_actions) or (highest_move is None) :
                high_prob_actions.append(highest_move)
            
        return high_prob_actions


    # Calculate each player's best spot
    # Return best moves of two models 
    def _calculate_best_actions(self, parent_turn) :
        if self._is_leaf() :
            return []

        best_child = None
        best_actions = []

        # Black : Find maximum
        if (parent_turn is BLACK) :
            max = MIN_VALUE - 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) > max) :
                    max = child._get_mean_value(parent_turn)
                    best_child = child
        # White : Find minumum
        elif (parent_turn is WHITE) :
            min = MAX_VALUE + 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) < min) :
                    min = child._get_mean_value(parent_turn)
                    best_child = child

        if (best_child is None) :
            print("Error : why search result is 0 in _calculate_best_actions")
            return []
        
        best_actions.append(best_child.action)
        best_actions.extend(best_child._calculate_best_actions(self.turn))

        return best_actions
    

    def select_best_child(self, parent_turn) :
        # - Parent will evaluate child's value
        if self._is_leaf() :
            print("Error : is_leaf in select_best_child")
            return None

        best_child = None

        # Black : Find maximum
        if (parent_turn is BLACK) :
            ucb_score_max = MIN_VALUE

            for child in self.children :
                if (child._ucb_score(parent_turn) > ucb_score_max) :
                    ucb_score_max = child._ucb_score(parent_turn)
                    best_child = child

        # White : Find minumum
        elif (parent_turn is WHITE) :
            ucb_score_min = MAX_VALUE

            for child in self.children :
                if (child._ucb_score(parent_turn) < ucb_score_min) :
                    ucb_score_min = child._ucb_score(parent_turn)
                    best_child = child

        if (best_child is None) :
            print("Error : why search result is 0 in select_best_child")
            return None

        return best_child


    def select_best_child_without_ucb(self, parent_turn) :
        # - Parent will evaluate child's value
        if self._is_leaf() :
            print("Error : is_leaf in select_best_child_without_ucb")
            return None, 0.0

        best_child = None

        # Black : Find maximum
        if (parent_turn is BLACK) :
            max = MIN_VALUE - 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) > max) :
                    max = child._get_mean_value(parent_turn)
                    best_child = child
        # White : Find minumum
        elif (parent_turn is WHITE) :
            min = MAX_VALUE + 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) < min) :
                    min = child._get_mean_value(parent_turn)
                    best_child = child

        if (best_child is None) :
            print("Error : why search result is 0 in select_best_child")
            return None, 0.0

        best_actions = self._calculate_best_actions(parent_turn)
    
        return best_child.action, best_actions


    def select_random_child(self) :
        if self._is_leaf() :
            print("Error : is_leaf in select_random_child")
            return None

        return rnd.choice(self.children)


    ## select best leaf, with ucb
    def select_best_leaf(self, parent_turn) :
        if self._is_leaf() :
            return self

        best_child = None
        
        best_child = self.select_best_child(parent_turn)

        if (best_child is None) :
            print("Error, best_child is None [ select_best_leaf ]")
            return self

        # At next recursive, current child will become parents.
        # So, reverse parent_turn value
        return best_child.select_best_leaf(parent_turn * -1)


    def select_best_leaf_with_random(self, parent_turn, random_prob=RANDOM_MOVE_PROB) :
        if self._is_leaf() is True :
            return self

        best_child = None

        # - Search child randomly with probability of random_prob
        if (rnd.random() < RANDOM_MOVE_PROB) :
            best_child = self.select_random_child()
        else :
            best_child = self.select_best_child(parent_turn)

        if (best_child is None) :
            print("Error, best child is None [ select_best_leaf_with_random ] ")
            return self

       # At next recursive, current child will become parents.
        # So, reverse parent_turn value
        return best_child.select_best_leaf_with_random(parent_turn * -1)


    def expand(self) :
        if (self._is_leaf() is False) :
            print("Error : expand node is not leaf")
            return self.select_best_leaf_with_random(self.turn)

        # If enter the end of game, do not expand.
        if (self.action is None) and (self._is_root() is False) :
            return self

        self._calculate_possible_actions()
        self._calculate_action_spot_prob()

        if (self._is_root() is True)  :
            actions_space = self.possible_actions
        else :
            actions_space = self.high_prob_actions

        if (self.turn == WHITE) :
            next_move_count = self.move_count + 1
        else :
            next_move_count = self.move_count

        for action in actions_space :
            new_GameState = self.GameState.copy()
            
            set_move(new_GameState, action)

            if (new_GameState.is_done(ADDITIONAL_SEARCH_COUNT_DURING_MCTS) is True) :
                del new_GameState
                continue

            next_turn = self.turn * -1

            child_spot_prob = self._calculate_spot_prob_from_action(action)

            new_mcts_node = MCTS_Node(action, self, child_spot_prob, next_turn, next_move_count, self.Model, new_GameState)
            self.children.append(new_mcts_node)

        return self.select_best_leaf_with_random(self.turn)


    def rollout(self, t_max=1) :
        # double check, calculate init value 
        self._calculate_action_spot_prob()

        # During expand, create new MCTS_Node includes calculating its own winning probability
        total_reward = self.node_init_value
        
        return total_reward


    def propagate(self, update_value, update_list, p_value=0.1) :
        if (float(update_value) is 0.0) :
            return

        self.value_sum += update_value
        self.times_visited += 1

        if (self.value_min > update_value) :
            self.value_min = update_value

        if (self.value_max < update_value) :
            self.value_max = update_value


        if not self._is_root() :
            update_list.append(self.action)
            self.parent.propagate(update_value, update_list)
        #else :
            #print(update_list)
            #print(self.best_actions)
            #print(self.value_min)

        return


    def delete(self) :
        if (len(self.children) is not 0) :
            for child in self.children :
                child.delete()

        del self.GameState
        del self

        return


    # - caller might delete [spot_count]
    def get_spot_count(self) :
        spot_count = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=int)

        for child in self.children :
            action_x, action_y = child.action

            spot_count[action_y][action_x] = child.times_visited

        return spot_count


    # - caller might delete [spot_prob]
    def get_spot_prob(self) :
        spot_prob = np.zeros([SQUARE_BOARD_SIZE + 1])
        
        summarized = 0.0

        for child in self.children :
            action_x, action_y = child.action
            action_idx = action_y * BOARD_SIZE + action_x

            spot_prob[action_idx] = child.times_visited
            summarized += child.times_visited

        if (summarized == 0.0) :
            summarized = 1.0

        return (spot_prob / summarized)


    # - caller might delete [win_prob_map]
    def get_win_prob(self) :
        win_prob_map = np.zeros([BOARD_SIZE, BOARD_SIZE])

        for child in self.children :
            action = child.action
            win_prob = child._get_mean_value(self.turn)

            action_x, action_y = action
            win_prob_map[action_y][action_x] = win_prob

        return win_prob_map


    # - caller might delete [scores]
    def get_ucb_scores(self) :
        scores = np.zeros([BOARD_SIZE, BOARD_SIZE])

        for child in self.children :
            action = child.action
            ucb = child._ucb_score(self.turn)

            action_x, action_y = action
            scores[action_y][action_x] = ucb

        return scores


    # Finalize function.
    # - Each model's will call this function at the end of the get_move() function
    def summary(self) :
        # Figure 2 : UCB Score
        scores = self.get_ucb_scores()
        #Draw_Win_Spot(scores, figure_num=2, title="UCB Score")

        # Figure 1 : spot's win probability 0(Lose) ~ 1(WIN)
        win_prob_map = self.get_win_prob()
        Draw_Win_Spot(win_prob_map, figure_num=1, title="Win probabilities")

        # Figure 4 : spot's importance - based on MCTS's search time
        spot_prob = self.get_spot_prob()
        #Draw_Win_Spot(spot_prob, figure_num=4, title="Spot probabilities")

        # Show spot / win probablitiy 
        print(np.around(self.get_spot_prob(), decimals=3))
        print(np.around(self.get_win_prob(), decimals=3))

        del win_prob_map
        del scores
        del spot_prob

        return

    # From current state, get maximum win probability
    def get_max_win_prob(self) :
        return self._get_mean_value(self.turn)


class Root_Node(MCTS_Node) :
    
    def __init__(self, Model, GameState, turn, move_count) :
        self.parent = None
        self.action = None # Root node's previous action is None
        self.prior_prob = None

        self.turn = turn # Me : 1 , Enemy = -1 -> 1 at root (my turn)
        self.move_count = move_count # Increased only white -> black
        self.Model = Model
        self.GameState = GameState

        self.value_sum = 0.0
        self.times_visited = 0

        self.children = []

        self.possible_actions_calculated = False
        self.action_spot_prob_calculated = False

    def play_mcts(self, n_iters=MCTS_N_ITERS) :
        for i in range(0, n_iters) :
            if (i % 512) == 0 :
                print("mcts iteration : ", i)
            
            best_leaf = self.select_best_leaf_with_random(self.turn)

            if (best_leaf.GameState.is_done(ADDITIONAL_SEARCH_COUNT_DURING_MCTS) is True) :
                print(" !! warning : access IS_DONE !!")
                best_leaf.propagate(0.0)

            else :
                new_best_leaf = best_leaf.expand()
                total_rewards = new_best_leaf.rollout()
                new_best_leaf.propagate(total_rewards, [])
