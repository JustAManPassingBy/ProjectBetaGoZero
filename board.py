import numpy as np
import copy

# UNFORTUNATELY, This function / file is deprecated

QUEUE_SIZE = 4
PARTIAL_OBSERVABILITY = 4

class Board():

    def __init__(self) :
        self.board_queue = []
        self.num_board_queue = 0
        self.moves = 0

    def get_parameters(self) :
        return self.board_queue, self.num_board_queue, self.moves

    def parameters_after_copy(self, board_queue, num_board_queue, moves) :
        self.board_queue = copy.deepcopy(board_queue)
        self.num_board_queue = copy.deepcopy(num_board_queue)
        self.moves = copy.deepcopy(moves)
    
    ### Board Update ###
    def update(self, inputs) :
        self.board_queue.append(inputs)

        self.moves += 1

        if (self.num_board_queue == QUEUE_SIZE) :
            # Delete first inputs
            self.board_queue = self.board_queue[1:]
        else :
            self.num_board_queue += 1

    # Player : me = 1 , enemy = -1
    def update_one_move(self, move, player, board_update=True) :
        if (self.num_board_queue is 0) :
            print("Error : empty queues at update_one_move")
            return
        
        # calculate move (if required)
        move_y = int(move / 19)
        move_x = int(move % 19)

        self.moves += 1

        # Update board queue
        if (board_update is True) :
            copy_latest_board = copy.deepcopy(self.board_queue[self.num_board_queue - 1])
            copy_latest_board[move_y][move_x] = player

            self.update(copy_latest_board)
        # Update only last queue
        else :
            self.board_queue[self.num_board_queue - 1][move_y][move_x] = player
        

    def unroll_one_move(self, move) :
        if (self.num_board_queue is 0) :
            print("Error : empty queues at unroll_one_move")
            return

        self.moves -= 1
        
        # calculate move (if required)
        move_y = int(move / 19)
        move_x = int(move % 19)

        self.board_queue[self.num_board_queue - 1][move_y][move_x] = 0


    def get_all(self) :
        # DeepCopy
        ret_array = np.zeros([19, 19, 4])

        for i in range(0, 19) :
            for j in range(0, 19) :
                for k in range(3, -1, -1) :
                    queue_idx = self.num_board_queue - 4 + k
                    if (queue_idx < 0) :
                        ret_array[i][j][k] = 0
                    else :
                        ret_array[i][j][k] = self.board_queue[k - (4 - self.num_board_queue)][i][j]

        return ret_array

    def get_latest(self) :
        # Not a deep copy
        return self.board_queue[self.num_board_queue - 1]


    def is_done(self) :
        return (self.moves > (361 - 40))

    def safe_copy(self) :
        new_board = Board()
        board_queue, num_board_queue , moves = self.get_parameters()
        new_board.parameters_after_copy(board_queue, num_board_queue, moves)

        return new_board

        
        
        
