import numpy as np
import math
import random

import game

from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT
from definitions import PARTIAL_OBSERVABILITY, MCTS_ADDITIONAL_SEARCH
from definitions import MAX_TURN
from definitions import MCTS_N_ITERS

from record_functions import Draw_Win_Spot

class MCTS_Node():
    
    def __init__(self, action, parent, turn, move_count, Model, GameState) :
        self.parent = parent # None if Root node
        self.action = action
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

    def _get_mean_value(self) :
        self._calculate_action_spot_prob()

        '''
        return self.value_sum / self.times_visited # Use Average
        '''

        return self.value_min

    def _calculate_possible_actions(self) :

        if (self.possible_actions_calculated is False) :
            self.possible_actions = self.GameState.get_legal_moves(include_eyes=True)

            self._calculate_action_spot_prob()

            self.high_prob_actions = self._get_high_prob_actions()

            self.possible_actions_calculated = True

    def _calculate_action_spot_prob(self,
                                    opposite=False) :

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

    def _get_high_prob_actions(self, num_actions=10) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        for _ in range(num_actions) :
            find = False
            times = 0

            while (find is False) :
                times += 1

                highest = self.action_prob.flatten().argmax()
                prev_highest_value = self.action_prob[0][highest]
                self.action_prob[0][highest] = MIN_VALUE

                highest_y = int(highest / BOARD_SIZE)
                highest_x = highest % BOARD_SIZE

                highest_move = (highest_y , highest_x)
                
                if (prev_highest_value <= MIN_VALUE) :
                    find = True

                if (highest_move in self.possible_actions) :
                    find = True
                    high_prob_actions.append(highest_move)


        return high_prob_actions

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


    def select_best_leaf(self) :
        if self._is_leaf() :
            return self

        ucb_score_max = MIN_VALUE - 1.0
        
        best_child = None

        for child in self.children :
            if (child._ucb_score() > ucb_score_max) :
                ucb_score_max = child._ucb_score()
                best_child = child

        if (best_child is None) :
            return self

        return best_child.select_best_leaf()


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


    def expand(self) :
        if (len(self.children) is not 0) :
            print("Error : expand node is not leaf")
            return self.select_best_leaf()

        self._calculate_possible_actions()
        self._calculate_action_spot_prob()

        if (self._is_root() is True) :
            actions_space = self.possible_actions
        else :
            actions_space = self.high_prob_actions

        for action in actions_space :
            new_GameState = self.GameState.copy()
            new_GameState.do_move(action) 

            if (self.turn == WHITE) :
                next_move_count = self.move_count + 1
            else :
                next_move_count = self.move_count

            if (new_GameState.is_done(MCTS_ADDITIONAL_SEARCH) is True) or (next_move_count >= MAX_TURN):
                del new_GameState
                continue

            self.children.append(MCTS_Node(action, self, self.turn * -1, next_move_count, self.Model, new_GameState))

        return self.select_best_leaf()


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
        # Propagate : Latest results will reflect much precise prediction
        # Therefore, new_value = value_old * p + update_value * (1 - p)  
        # new_value = (self.value_sum * p_value) + (update_value * (1 - p_value))
        #self.value_sum = new_value 
        
        if (float(update_value) is 0.0) :
            return

        self.value_sum += update_value
        self.times_visited += 1

        if (self.value_min > update_value) :
            self.value_min = update_value

        if not self._is_root() :
            self.parent.propagate(update_value)

    def delete(self) :
        if (len(self.children) is not 0) :
            for child in self.children :
                child.delete()

        del self.GameState
        del self

    def get_spot_prob(self) :
        spot_prob = np.zeros([BOARD_SIZE, BOARD_SIZE])
        
        summarized = 0.0

        for child in self.children :
            action_x, action_y = child.action

            spot_prob[action_y][action_x] = child.times_visited
            summarized += child.times_visited

        if (summarized == 0.0) :
            summarized = 1.0

        return (spot_prob / summarized)

    def get_spot_with_prob(self) :
        spot_list = []
        spot_prob_list = []

        summarized = 0.0

        for child in self.children :
            spot_list.append(child.action)
            spot_prob_list.append(child.times_visited)

            summarized += child.times_visited

        if (summarized == 0.0) :
            summarized = 1.0

        for i in range(0, len(spot_prob_list)) :
            spot_prob_list[i] /= summarized

        return spot_list, spot_prob_list

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

    # From current state, get maximum win probability
    #  - Pseuco code
    # for spot in Board :
    #    max(max_spot.win_prob(), spot.win_prob())
    # return max_spot.win_prob()
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
            
            best_leaf = self.select_best_leaf()

            if (best_leaf.GameState.is_done(MCTS_ADDITIONAL_SEARCH) is True) :
                print(" !! warning : access IS_DONE !!")
                best_leaf.propagate(0.0)

            else :
                new_best_leaf = best_leaf.expand()
                total_rewards = new_best_leaf.rollout()
                new_best_leaf.propagate(total_rewards)
