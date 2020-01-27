import numpy as np
import math
import random

import game

from matplotlib import pyplot as plt

from definitions import PASS_MOVE, BLACK, EMPTY, WHITE, KOMI
from definitions import MAX_VALUE, MIN_VALUE
from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT
from definitions import PARTIAL_OBSERVABILITY, MCTS_ADDITIONAL_SEARCH

class MCTS_Node():
    
    def __init__(self, action, parent, turn, Model, GameState) :
        self.parent = parent # None if Root node
        self.action = action
        self.turn = turn     # BLACK : 1 , WHITE = -1
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

        return self.value_sum / self.times_visited

    def _calculate_possible_actions(self) :

        if (self.possible_actions_calculated is False) :
            self.possible_actions = self.GameState.get_legal_moves(include_eyes=True)

            self._calculate_action_spot_prob()

            self.high_prob_actions = self._get_high_prob_actions()

            self.possible_actions_calculated = True

    def _calculate_action_spot_prob(self) :
        if (self.action_spot_prob_calculated is False) :
            # (1, 361) / Scalar
            self.action_prob , self.node_init_value = self.Model.predict_func(self.GameState.show_4_latest_boards()) # spot_prob / win_prob
            self.node_init_value = np.asscalar(self.node_init_value)

            self.value_sum = self.node_init_value
            self.times_visited += 1

            self.action_spot_prob_calculated = True

    def _get_high_prob_actions(self, num_actions=10) :
        self._calculate_action_spot_prob()
        
        high_prob_actions = []

        for i in range(num_actions) :
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

        max_vaule = MIN_VALUE
        
        for child in self.children :
            if (child._get_mean_value() > max_vaule) :
                max_vaule = child._get_mean_value()
                best_child = child

        return best_child.action

    # UCB 1
    def _ucb_score(self, scale=0.25, max_value=MAX_VALUE) :
        if (self.times_visited < 1) :
            return max_value
        
        ucb = math.sqrt(2 * math.log(self.parent.times_visited) / self.times_visited)

        return self._get_mean_value() + scale * ucb


    def select_best_leaf(self) :
        if self._is_leaf() :
            return self

        ucb_score_max = MIN_VALUE
        
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
            return self

        ucb_score_max = MIN_VALUE

        for child in self.children :
            if (child._ucb_score() > ucb_score_max) :
                ucb_score_max = child._ucb_score()
                best_child = child

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

        num_search = 0

        for action in actions_space :
            new_GameState = self.GameState.copy()
            new_GameState.do_move(action) 

            if (new_GameState.is_done(MCTS_ADDITIONAL_SEARCH) is True) :
                del new_GameState

                continue

            self.children.append(MCTS_Node(action, self, self.turn * -1, self.Model, new_GameState))
            num_search += 1

        if (num_search is 0) :
            print("Error num search is 0")

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
            action_y, action_x = child.action

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

            action_y, action_x = action
            win_prob_map[action_y][action_x] = win_prob

        return win_prob_map

    # Should delete "scores"
    def get_ucb_scores(self) :
        scores = np.zeros([BOARD_SIZE, BOARD_SIZE])

        for child in self.children :
            action = child.action
            ucb = child._ucb_score()

            action_y, action_x = action
            scores[action_y][action_x] = ucb

        return scores

    def summary(self) :
        # Figure 2 : UCB Score
        plt.figure(2, clear=True)

        scores = self.get_ucb_scores()

        #title_string = str("UCB Scores : (MAX/MIN) " + str(max(scores)) + " / " + str(min(scores)))
        
        plt.imshow(scores)
        plt.title("UCB Score")
        plt.colorbar()

        plt.draw()
        plt.pause(0.1)

        # Figure 1 : spot's win probability -1(Lose) ~ 1(WIN)
        plt.figure(1, clear=True)

        win_prob_map = self.get_win_prob()

        #title_string = str("win_values : (MAX/MIN) " + str(max(win_prob_map)) + " / " + str(min(win_prob_map)))

        plt.imshow(win_prob_map)
        plt.title("Win probabilities")
        plt.colorbar()

        plt.draw()
        plt.pause(0.1)

        # Figure 4 : spot's importance - based on MCTS's search time
        plt.figure(4, clear=True)

        spot_prob = self.get_spot_prob()
        
        plt.imshow(spot_prob)
        plt.title("Spot probabilities")
        plt.colorbar()

        plt.draw()
        plt.pause(0.1)

        del win_prob_map
        del scores
        del spot_prob

    # From current state, get maximum win probability
    #  - Pseuco code
    # for spot in Board :
    #    max(max_spot.win_prob(), spot.win_prob())
    # return max_spot.win_prob()
    def get_max_win_prob(self) :
        max_win_prob = MIN_VALUE

        for child in self.children :
            if (child._get_mean_value() > max_win_prob) :
                max_win_prob = child._get_mean_value()

        return max_win_prob

class Root_Node(MCTS_Node) :
    
    def __init__(self, Model, GameState, turn) :
        self.parent = None
        self.action = None
        self.turn = turn # Me : 1 , Enemy = -1 -> 1 at root (my turn)
        self.Model = Model
        self.GameState = GameState

        self.value_sum = 0.0
        self.times_visited = 0

        self.children = []

        self.possible_actions_calculated = False
        self.action_spot_prob_calculated = False

    def play_mcts(self, n_iters=140) :
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
