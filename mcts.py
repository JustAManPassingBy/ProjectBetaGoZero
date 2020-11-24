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
from definitions import TRAIN_COUNTS_TO_SKIP_NONE_MOVE

from record_functions import Draw_Win_Spot

from game_hub import *

# Each node means specific move
#            (Root Node)
#           Initial state
#             (Turn : 1)
#             /       \
#         (Node 1)   (Node 2)
#          Move 1     Move 2
#       (Turn : -1) (Turn : -1)
#          /  \
#           ... 

class MCTS_Node():
    
    # External program will not create class from this.
    # Instead, they will use Root_Node()
    def __init__(self, action, parent, prior_prob, turn, move_count, game_count, Model, GameState, debug_mode=0) :
        self.parent = parent # None if Root node
        self.action = action # parent node's action "a"
        self.turn = turn     # BLACK : 1 , WHITE = -1
        self.move_count = move_count
        self.game_count = game_count
        self.Model = Model

        self.GameState = GameState

        self.value_sum = 0.0     # This value is same with Q(s, a), where s, a is parent node's state & action
        self.times_visited = 0   # This value is same with N(s, a), where s, a is parent node's state & action

        self.children = []
        self.node_is_leaf = True

        self.possible_actions_calculated = False
        self.initial_node_value_calculated = False

        self.black_best_action = None
        self.white_best_action = None

        self.prior_prob = prior_prob # This value is P(s, a), where s, a is parent node's state & action
        
        self.debug_mode = debug_mode

    def _is_leaf(self) :
        # return self.node_is_leaf
        return len(self.children) is 0


    def _is_root(self) :
        return self.parent is None

    
    # Standard format for node's value
    def get_node_value(self, turn) :
        return self._get_mean_value(turn)


    # Calculate Q Value of current node Q(s, a)
    # Note : This value is meaningless at root node.
    def _get_mean_value(self, turn) :
        self.calculate_init_node_value()

        return self.value_sum / self.times_visited # Use Average


    def _get_high_value(self, turn) :
        self.calculate_init_node_value()
        
        if (turn is BLACK) :
            val = self.value_max
        else :
            val = self.value_min

        return val


    # Calculate UCB-included Q Value Q'(s, a) = Q(s, a) + U(s, a)
    def _ucb_score(self, turn, scale=1.5) :  
        self.calculate_init_node_value()

        # U(s, a) = scale * P(s, a) / (1 + N(s, a))
        ucb = self.prior_prob / (1 + self.times_visited)
        
        # White's ucb = ucb * -1 (turn)
        # Black's ucb = ucb * 1  (turn)
        scaled_ucb = ucb * scale * turn

        return self.get_node_value(turn) + scaled_ucb


    # Calculate Possible action spots A, a ∈ A
    def calculate_possible_actions(self) :
        if (self.possible_actions_calculated is False) :
            self.possible_actions = get_valid_states(self.GameState)

            if (self.game_count >= TRAIN_COUNTS_TO_SKIP_NONE_MOVE) :
                self.possible_actions.append(None) # None means "skip to plate stone"

            # Below function must assert "Calculating initial node value" 
            self.high_prob_actions = self._get_high_prob_actions()

            self.possible_actions_calculated = True


    # Calculate initial node's value
    # It includes action / spot probabilities, node's value, times_visited, etc..
    def calculate_init_node_value(self,
                                  opposite=False) :
        if (self.initial_node_value_calculated is False) :
            self.action_prob , self.node_init_value = self.Model.predict_func(self.GameState.show_4_latest_boards(), self.move_count) # spot_prob / win_prob

            self.node_init_value = np.asscalar(self.node_init_value)
            self.action_prob = self.action_prob.flatten()

            self.value_sum = self.node_init_value
            self.times_visited += 1

            self.value_min = self.node_init_value
            self.value_max = self.node_init_value

            self.black_best_actions = [self.action]
            self.white_best_actions = [self.action]

            self.initial_node_value_calculated = True


    # Calculate π(a) [pi(a)]
    def _calculate_spot_prob_from_action(self, action) :
        if (action is None) :
            return self.action_prob[BOARD_SIZE]

        (action_x, action_y) = action
        action_idx = action_x + action_y * BOARD_SIZE

        self.calculate_init_node_value()

        return self.action_prob[action_idx]


    ##  Functions to get high probability actions (Calculate A', which a' ∈ π)
    def _get_high_prob_actions(self, num_actions=12) :
        self.calculate_init_node_value()
        
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
        self.calculate_init_node_value()
        
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


    def _get_high_prob_actions_from_max_value(self, limit=0.1) :
        self.calculate_init_node_value()
        
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
    # Return array of actions(moves) calculated by two models 
    def _calculate_best_actions(self, parent_turn) :
        if self._is_leaf() :
            return []

        best_child = None
        best_actions = []

        # Black : Q(s, a) is better if this value is closer to 1
        if (parent_turn is BLACK) :
            max = MIN_VALUE - 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) > max) :
                    max = child._get_mean_value(parent_turn)
                    best_child = child
        # White : Q(s, a) is better if this value is closer to 0 (or -1)
        elif (parent_turn is WHITE) :
            min = MAX_VALUE + 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) < min) :
                    min = child._get_mean_value(parent_turn)
                    best_child = child

        if (best_child is None) :
            print("Error : best child is None in _calculate_best_actions")
            return []
        
        # Near future's move will be located in frone of the list
        best_actions.append(best_child.action)
        best_actions.extend(best_child._calculate_best_actions(self.turn))

        return best_actions
    

    def select_best_child(self, parent_turn) :
        # - In perspective of parent, choose child.
        #    * If parent turn is black, best child means _ucb_score() (Q'(s, a)) is high
        #    * If parent turn is white, best child means _ucb_score() (Q'(s, a)) is low
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

        if (isinstance(best_child, MCTS_Node) is False) :
            print("Error : best child is not class in select_best_child")
            print(best_child)

        if (self.debug_mode == 2) :
            ucb_score_map = self.get_ucb_scores()
            win_score_map = self.get_win_prob()

            print("[Select_child_with_ucb] next move :", best_child.action)
            print(" --- ucb ---")
            print(ucb_score_map)
            print(" --- win ---")
            print(win_score_map)

            del ucb_score_map
            del win_score_map

        return best_child


    def select_best_child_without_ucb(self, parent_turn) :
        # - In perspective of parent, choose child.
        #    * If parent turn is black, best child means _mean_value (Q(s, a)) is closer to 1
        #    * If parent turn is white, best child means _mean_value (Q(s, a)) is closer to 0 or -1
        if self._is_leaf() :
            print("Error : is_leaf in select_best_child_without_ucb")
            return None

        best_child = None

        # Black : Q(s, a) is better if this value is closer to 1
        if (parent_turn is BLACK) :
            max = MIN_VALUE - 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) > max) :
                    max = child._get_mean_value(parent_turn)
                    best_child = child
        # White : Q(s, a) is better if this value is closer to 0 or -1
        elif (parent_turn is WHITE) :
            min = MAX_VALUE + 1

            for child in self.children :
                if (child._get_mean_value(parent_turn) < min) :
                    min = child._get_mean_value(parent_turn)
                    best_child = child

        if (isinstance(best_child, MCTS_Node) is False) :
            print("Error : best child is not class in select_best_child_without_ucb")
            print(best_child)

        # best_actions = self._calculate_best_actions(parent_turn)
    
        if (self.debug_mode == 2) :
            mean_value_map = self.get_win_prob()

            print("[Select_child_wo_ucb] next move :", best_child.action)
            print(" --- win ---")
            print(mean_value_map)

            del mean_value_map

        return best_child


    def select_random_child(self) :
        if self._is_leaf() :
            print("Error : is_leaf in select_random_child")
            return None

        if len(self.children) is 0 :
            return None

        picked_child = rnd.choice(self.children)

        if (isinstance(picked_child, MCTS_Node) is False) :
            print("Error : picked_child is not class in select_random_child")
            print(picked_child)

        if (self.debug_mode == 2) :
            print("[Random pick] action :", picked_child.action)

        return picked_child


    ## select best leaf, with ucb
    def select_best_leaf(self, parent_turn) :
        if self._is_leaf() :
            return self

        best_child = self.select_best_child(parent_turn)

        # At next recursive, current node becomes parent.
        # So, reverse parent_turn value
        return best_child.select_best_leaf(parent_turn * -1)


    def select_best_leaf_with_random(self, parent_turn, search_depth, random_prob=RANDOM_MOVE_PROB) :
        if self._is_leaf() is True :
            return self, search_depth

        if (self.debug_mode == 2) :
            print("[select_best_leaf_with_random] turn : ", self.turn)

        # - Search child randomly with probability of random_prob
        if (rnd.random() < RANDOM_MOVE_PROB) :
            best_child = self.select_random_child()
        else :
            best_child = self.select_best_child(parent_turn)

       # At next recursive, current child will become parents.
        # So, reverse parent_turn value
        return best_child.select_best_leaf_with_random(parent_turn * -1, search_depth + 1)


    def expand(self) :
        if (self._is_leaf() is False) :
            print("Error : expand node is not leaf")
            return self.select_best_leaf_with_random(self.turn, 1)

        # If enter the end of game, do not expand.
        if (self.action is None) and (self._is_root() is False) :
            return self, 1

        self.calculate_possible_actions()

        if (self._is_root() is True)  :
            actions_space = self.possible_actions
        else :
            actions_space = self.high_prob_actions

        if (self.turn == WHITE) :
            next_move_count = self.move_count + 1
        else :
            next_move_count = self.move_count

        if (len(actions_space) < 1) :
            print("[Error - expand]  : action space is less than 1")

        for action in actions_space :
            new_GameState = self.GameState.copy()
            
            set_move(new_GameState, action)

            if (new_GameState.is_done(ADDITIONAL_SEARCH_COUNT_DURING_MCTS) is True) :
                del new_GameState
                continue

            next_turn = self.turn * -1
            child_spot_prob = self._calculate_spot_prob_from_action(action)

            new_mcts_node = MCTS_Node(action, self, child_spot_prob, next_turn, next_move_count, self.game_count, self.Model, new_GameState, debug_mode=self.debug_mode)
            
            if (isinstance(new_mcts_node, MCTS_Node) is False) :
                print("Error : new_mcts_node is not object of MCTS Node")
                print(new_mcts_node)

            self.children.append(new_mcts_node)

        # Debug purpose
        if (len(self.children) < 1) :
            print("[expand] num_children : ", len(self.children))
            print(actions_space)

        return self.select_best_leaf_with_random(self.turn, 1)


    def rollout(self, t_max=1) :
        # double check, calculate init value 
        self.calculate_init_node_value()

        # During expand, create new MCTS_Node includes calculating its own winning probability
        # Maybe play some mini game?
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
        else :
            if (self.debug_mode == 2) :
                print("After propagagte")

                ucb_scores = self.get_ucb_scores()
                win_prob = self.get_win_prob()

                print(" --- ucb --- ")
                print(ucb_scores)

                print(" --- win --- ")
                print(win_prob)

                del ucb_scores
                del win_prob

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
            if (child.action is None) :
                # TBD : We add visible tools for "None(not place stone)"
                continue

            action_x, action_y = child.action

            spot_count[action_y][action_x] = child.times_visited

        return spot_count


    # - caller might delete [spot_prob]
    def get_spot_prob(self) :
        spot_prob = np.zeros([SQUARE_BOARD_SIZE + 1])
        
        summarized = 0.0

        for child in self.children :
            if (child.action is None) :
                action_idx = SQUARE_BOARD_SIZE
            else :
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
            if (child.action is None) :
                # TBD : We add visible tools for "None(not place stone)"
                continue

            action = child.action
            win_prob = child._get_mean_value(self.turn)

            action_x, action_y = action
            win_prob_map[action_y][action_x] = win_prob

        return np.around(win_prob_map, decimals=3)


    # - caller might delete [scores]
    def get_ucb_scores(self) :
        scores = np.zeros([BOARD_SIZE, BOARD_SIZE])

        for child in self.children :
            if (child.action is None) :
                # TBD : We add visible tools for "None(not place stone)"
                continue

            action = child.action
            ucb = child._ucb_score(self.turn)

            action_x, action_y = action
            scores[action_y][action_x] = ucb

        return np.around(scores, decimals=3)


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
        print(np.around(self.get_spot_count()))

        del win_prob_map
        del scores
        del spot_prob

        return

    def get_best_move_and_actions(self) :
        best_child = self.select_best_child_without_ucb(self.turn)

        if (best_child is None) :
            best_move = None
            best_move_win_prob = best_child._get_mean_value(self.turn)
            best_actions = None
        else :
            best_move = best_child.action
            best_move_win_prob = best_child._get_mean_value(self.turn)
            best_actions = self._calculate_best_actions(self.turn)

        return best_move, best_move_win_prob, best_actions



class Root_Node(MCTS_Node) :
    
    def __init__(self, Model, GameState, turn, move_count, game_count, debug_mode=0) :
        self.parent = None      # Root node has no parent
        self.action = None      # Root node's previous action is None
        self.prior_prob = None  # Root node has no prior action or prior prob

        self.turn = turn             # BLACK = 1 / WHITE = -1
        self.move_count = move_count # Increased only white -> black
        self.game_count = game_count # Current Train is #th train.
        self.Model = Model
        self.GameState = GameState

        self.value_sum = 0.0         # Warn : At Root node, using this value is meaningless
        self.times_visited = 0       # Warn : At Root node, using this value is meaningless

        self.children = []
        self.node_is_leaf = True

        self.possible_actions_calculated = False
        self.initial_node_value_calculated = False

        self.debug_mode = debug_mode

        # Debug purpose initialization
        if (self.debug_mode == 2) :
            self.search_depth_summary = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    def play_mcts(self, n_iters=MCTS_N_ITERS) :
        for i in range(0, n_iters) :
            if (i % 16) == 0 :
                print("mcts iteration : ", i)
            
            best_leaf, search_depth = self.select_best_leaf_with_random(self.turn, 0)
            
            if (best_leaf is None) :
                print("[Error] play_mcts : best_leaf is none")

            if (self.debug_mode == 2) :
                if (search_depth > 10) :
                    search_depth = 9

                self.search_depth_summary[search_depth] += 1
            
            if (isinstance(best_leaf, MCTS_Node) is False) :
                print("Error : new_mcts_node is not object of MCTS Node")

            if (best_leaf.GameState.is_done(ADDITIONAL_SEARCH_COUNT_DURING_MCTS) is True) :
                print(" !! warning : access IS_DONE !!")
                best_leaf.propagate(0.0, [])

            else :
                new_best_leaf, _ = best_leaf.expand()

                if (new_best_leaf is not None) :
                    total_rewards = new_best_leaf.rollout()
                    new_best_leaf.propagate(total_rewards, [])

        if (self.debug_mode == 2) :
            # Show search depth
            if (self.debug_mode == 2) :
                for i in range(1, 10) : # Search depth max count : 10
                    print("depth : " + str(i) + " | count : " + str(self.search_depth_summary[i]))

        return