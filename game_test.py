from go import Board, BoardError, View, clear, getch

# Model for test game
# UNFORTUNATELY, This function / file is deprecated

WHITE = -1
BLACK = 1
PASS_MOVE = None

BOARD_SIZE = 19

Go_Board = Board(BOARD_SIZE)

black_is_first = True
white_is_first = True

count = 0

while (1) :
    #x, y = input("x y : ").split(" ")    
    #action = (int(x), int(y))

    current_player = Go_Game.get_current_player()

    if current_player is BLACK :
        if black_is_first is True :
            black_x = 9
            black_y = 0
            black_move_dir = 1
            black_is_first = False
        else :
            black_y += black_move_dir

            if (black_y > 18) or (black_y < 0) :
                black_x += 1
                black_y -= black_move_dir
                black_move_dir = -black_move_dir
        y = black_y
        x = black_x
    else :
        if white_is_first is True :
            white_x = 8
            white_y = 0
            white_move_dir = 1
            white_is_first = False
        else :
            white_y += white_move_dir

            if (white_y > 18) or (white_y < 0) :
                white_x -= 1
                white_y -= white_move_dir
                white_move_dir = -white_move_dir

        y = white_y
        x = white_x

    count += 1

    if (count < 305) :
        action = (x, y)
    else :
        action = PASS_MOVE

    print(current_player, action)
    
    if Go_Game.do_move(action, color=current_player) is True :
        print("winner : ", str(Go_Game.get_winner()))

        board = Go_Game.show_result()
        print(board)
        break
    else :
        board = Go_Game.show_result()

        #print(board)
        

    
