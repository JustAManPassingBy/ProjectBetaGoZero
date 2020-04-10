import numpy as np
import math
import random

import game

from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, NUM_STONES_TO_END_GAME
from definitions import PARTIAL_OBSERVABILITY, ADDITIONAL_SEARCH_COUNT_DURING_MCTS
from definitions import MAX_TURN
from definitions import MCTS_N_ITERS

from record_functions import Draw_Win_Spot

# Each node means specific move
#            (Root Node)
#           Initial state
#             /       \
#         (Node 1)   (Node 2)
#          Move 1     Move 2
#          /  \
#           ... 

class MCTS_Node():
    
    def __init__(self, action, parent, turn, move_count, Model, GameState) :
        self.parent = parent # None if Root node
        self.action = action # Action (X, Y)
        self.turn = turn     # BLACK : 1 , WHITE = -1
        self.move_count = move_count
        self.Model = Model

        self.GameState = GameState

        self.value_sum = 0.0
        self.times_visited = 0

        self.children = []

        self.possible_actions_calculated = False
        self.action_spot_prob_calculated = False

    def _is_leaf(self) :
        return len(self.children) is 0

    def _is_root(self) :
        return self.parent is None

    def _get_mean_value(self,
                        multi_value_sum=0.6,
                        multi_value_min=0.4) :
        self._calculate_action_spot_prob()

        value_average =  self.value_sum / self.times_visited # Use Average

        return (multi_value_sum * value_average + multi_value_min * self.value_min)

    def _calculate_possible_actions(self) :

        if (self.possible_actions_calculated is False) :
            self.possible_actions = self.GameState.get_legal_moves(include_eyes=True)

            self._calculate_action_spot_prob()

            self.high_prob_actions = self._get_high_prob_actions_range_from_max()

            self.possible_actions_calculated = True

    def _calculate_action_spot_prob(self,
                                    opposite=True) :

        if (self.action_spot_prob_calculated is False) :
            # (1, 361) / Scalar
            if (opposite is True) or (self.turn != self.Model.get_team()):
                # Use opposite's action_prob only
                self.action_prob , _ = self.Model.opposite_Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) # spot_prob
                _ , self.node_init_value = self.Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) # win_prob
            else :
                self.action_prob , self.node_init_value = self.Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) # spot_prob / win_prob
            self.node_init_value = np.asscalar(self.node_init_value)

            self.value_sum = self.node_init_value
            self.value_min = self.node_init_value
            self.times_visited += 1

            self.action_spot_prob_calculated = True


    ##  Functions to get high probability actions
    def _get_high_prob_actions(self, num_actions=15) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        highest_sort = np.array(self.action_prob.flatten).argsort()

        for i in range(len(highest_sort)) :
            if (num_actions == 0) :
                break

            highest = self.action_prob[0][highest_sort[i]]

            # None : None action
            if (highest == SQUARE_BOARD_SIZE) :
                highest_move = None
            # Else : action
            else :
                highest_y = int(highest / BOARD_SIZE)
                highest_x = int(highest % BOARD_SIZE)

                # !! Move for action is (X, Y), not (Y, X)
                highest_move = (highest_x, highest_y)

            if (highest_move in self.possible_actions) and (highest_move is not None):
                num_actions -= 1
                high_prob_actions.append(highest_move)
            elif (highest_move is None) :
                num_actions -= 1

        return high_prob_actions

    def _get_high_prob_actions_exceed_threshold(self, threshold=float(1 / SQUARE_BOARD_SIZE + 1)) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        highest_list = np.array(self.action_prob.flatten())
        highest_sort = highest_list.argsort()

        for i in range(len(highest_sort)) :
            if (highest_list[highest_sort[i]] < threshold) :
                break

            highest = highest_list[highest_sort[i]]

            # None : None action
            if (highest == SQUARE_BOARD_SIZE) :
                highest_move = None
            # Else : action
            else :
                highest_y = int(highest / BOARD_SIZE)
                highest_x = int(highest % BOARD_SIZE)

                # !! Move for action is (X, Y), not (Y, X)
                highest_move = (highest_x, highest_y)

            if (highest_move in self.possible_actions) and (highest_move is not None):
                high_prob_actions.append(highest_move)
            
        return high_prob_actions

    def _get_high_prob_actions_range_from_max(self, limit=
    0.1) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        highest_list = np.array(self.action_prob.flatten())
        highest_sort = highest_list.argsort()

        for i in range(len(highest_sort)) :
            highest = highest_list[highest_sort[i]]

            if (i == 0) :
                action_limit = highest - limit

            if (highest < action_limit) :
                break

            # None : None action
            if (highest == SQUARE_BOARD_SIZE) :
                highest_move = None
            # Else : action
            else :
                highest_y = int(highest / BOARD_SIZE)
                highest_x = int(highest % BOARD_SIZE)

                # !! Move for action is (X, Y), not (Y, X)
                highest_move = (highest_x, highest_y)

            if (highest_move in self.possible_actions) and (highest_move is not None):
                high_prob_actions.append(highest_move)
            
        return high_prob_actions


    def select_best_child(self) :
        if self._is_leaf() :
            print("Error : is_leaf in select_best_child")
            return None

        best_child = None
        
        ucb_score_max = MIN_VALUE - 1

        for child in self.children :
            if (child._ucb_score() > ucb_score_max) :
                ucb_score_max = child._ucb_score()
                best_child = child

        if (best_child is None) :
            return None

        return best_child

    def select_best_child_without_ucb(self) :
        if self._is_leaf() :
            print("Error : is_leaf in select_best_child_without_ucb")
            return None

        max_vaule = MIN_VALUE - 1.0

        best_child = None
        
        for child in self.children :
                            
            if (child._get_mean_value() > max_vaule) :
                max_vaule = child._get_mean_value()
                best_child = child

        if (best_child is None) :
            return None

        return best_child.action

    # UCB 1
    def _ucb_score(self, scale=0.05, max_value=MAX_VALUE) :
        if (self.times_visited < 1) :
            return max_value
        
        ucb = math.sqrt(2 * math.log(self.parent.times_visited) / self.times_visited)

        return self._get_mean_value() + scale * ucb

    ## select best leaf, with ucb
    def select_best_leaf(self) :
        if self._is_leaf() :
            return self

        ucb_score_max = MIN_VALUE - 1.0
        
        best_child = None

        for child in self.children :
            if (child._ucb_score() > ucb_score_max) :
                ucb_score_max = child._ucb_score()
                best_child = child

        # Why not_leaf node has no child?
        if (best_child is None) :
            return self

        return best_child.select_best_leaf()

    # Depth should be start from 0
    def select_best_and_worst_leaf(self, layer_depth=0) :
        if (self._is_leaf()) :
            return self, layer_depth

        # Even, our turn
        if (layer_depth % 2) == 0 :
            ucb_score_max = MIN_VALUE - 1.0

            best_child = None

            for child in self.children :
                if (child._ucb_score() > ucb_score_max) :
                    ucb_score_max = child._ucb_score()
                    best_child = child

            if (best_child is None) :
                return self, layer_depth

        # Odd, enemy's turn
        else :
            ucb_score_min = MAX_VALUE + 1.0
            best_child = None

            for child in self.children :
                if (child._ucb_score() < ucb_score_min) :
                    ucb_score_min = child._ucb_score()
                    best_child = child

            if (best_child is None) :
                return self, layer_depth

        return best_child.select_best_and_worst_leaf(layer_depth=(layer_depth + 1))


    def expand(self, layer_depth=0) :
        if (len(self.children) is not 0) :
            print("Error : expand node is not leaf")
            return self.select_best_leaf()

        self._calculate_possible_actions()
        self._calculate_action_spot_prob()

        if (self._is_root() is True)  :
            actions_space = self.possible_actions
        else :
            actions_space = self.high_prob_actions

        for action in actions_space :
            # Skip if None
            if (action is None) :
                continue

            new_GameState = self.GameState.copy()
            new_GameState.do_move(action) 

            if (self.turn == WHITE) :
                next_move_count = self.move_count + 1
            else :
                next_move_count = self.move_count

            if (new_GameState.is_done(ADDITIONAL_SEARCH_COUNT_DURING_MCTS) is True) :
                del new_GameState
                continue

            self.children.append(MCTS_Node(action, self, self.turn * -1, next_move_count, self.Model, new_GameState))

        return self.select_best_and_worst_leaf(layer_depth=layer_depth)


    def rollout(self, t_max=1) :
        # double check, calculate init value 
        self._calculate_action_spot_prob()

        # During expand, create new MCTS_Node includes calculating its own winning probability
        total_reward = self.node_init_value

        '''
        total_reward = 0.0

        # Intialize
        move_prob = self.action_prob
        total_reward = self.node_init_value

        # Game state
        new_game = self.GameState.copy()

        # Turn
        cur_turn = self.turn
        
        # Simulate
        # With current model, predict current state (Update with win_prob)
        for i in range(0, t_max) :

            if (new_game.is_done() is True) :
                break

            move_prob = move_prob.flatten()

            action = -1
               
            while(action == -1) :
                max_prob = max(move_prob)

                action_idxs = np.argwhere(move_prob == max_prob)
                
                action = action_idxs.flatten()[0]
                
                action_y = int(action / BOARD_SIZE)
                action_x = action % BOARD_SIZE

                action_tuple = (action_y, action_x)
                
                if (new_game.is_legal(action_tuple) is False) :
                    move_prob[action] = -2147483648.0
                    action = -1

            new_game.do_move(action_tuple, color=new_game.get_current_player())

            move_prob , reward = self.Model.predict_func(new_game.show_4_latest_boards()) # spot_prob / win_prob
        
            total_reward += reward
            cur_turn *= -1

        del new_game
        '''
        
        return total_reward


    def propagate(self, update_value, p_value=0.1) :
        if (float(update_value) is 0.0) :
            return

        self.value_sum += update_value
        self.times_visited += 1

        if (self.value_min > update_value) :
            self.value_min = update_value

        if not self._is_root() :
            self.parent.propagate(update_value)

        return

    def delete(self) :
        if (len(self.children) is not 0) :
            for child in self.children :
                child.delete()

        del self.GameState
        del self

        return

    def get_spot_count(self) :
        spot_count = np.zeros([BOARD_SIZE, BOARD_SIZE], dtype=int)

        for child in self.children :
            action_x, action_y = child.action

            spot_count[action_y][action_x] = child.times_visited

        return spot_count

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


    def get_win_prob(self) :
        win_prob_map = np.zeros([BOARD_SIZE, BOARD_SIZE])

        for child in self.children :
            action = child.action
            win_prob = child._get_mean_value()

            action_x, action_y = action
            win_prob_map[action_y][action_x] = win_prob

        return win_prob_map

    # Should delete "scores"
    def get_ucb_scores(self) :
        scores = np.zeros([BOARD_SIZE, BOARD_SIZE])

        for child in self.children :
            action = child.action
            ucb = child._ucb_score()

            action_x, action_y = action
            scores[action_y][action_x] = ucb

        return scores

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

        del win_prob_map
        del scores
        del spot_prob

        return

    # From current state, get maximum win probability
    def get_max_win_prob(self) :
        max_win_prob = MIN_VALUE - 1

        for child in self.children :
            if (child._get_mean_value() > max_win_prob) :
                max_win_prob = child._get_mean_value()

        return max_win_prob


class Root_Node(MCTS_Node) :
    
    def __init__(self, Model, GameState, turn, move_count) :
        self.parent = None
        self.action = None
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
            #if (i % 128) == 0 :
            #print("mcts iteration : ", i)
            
            best_leaf, depth = self.select_best_and_worst_leaf()

            if (best_leaf.GameState.is_done(ADDITIONAL_SEARCH_COUNT_DURING_MCTS) is True) :
                print(" !! warning : access IS_DONE !!")
                best_leaf.propagate(0.0)

            else :
                new_best_leaf, _ = best_leaf.expand(layer_depth=depth)
                total_rewards = new_best_leaf.rollout()
                new_best_leaf.propagate(total_rewards)
