import numpy as np
import copy
from matplotlib import pyplot as plt

from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT
from definitions import BLACK, WHITE
from definitions import MAX_VALUE, MIN_VALUE

# Draw plot with Board
def Draw_Plot(board, 
              figure_num=3, 
              pause_time=0.1,
              save_image=False,
              save_image_name=None) :

    plt.figure(figure_num, clear=True)

    for y in range(0, BOARD_SIZE) :
        for x in range(0, BOARD_SIZE) :
            if (board[y][x] == 0) :
                size = 0
            else :
                size = 400

            if (board[y][x] == BLACK) :
                color = '#000000'
            else :
                color = '#ffffff'

            # game.py save board[x, y], not board[y, x]
            plt.scatter(y, x, s=size, c=color, edgecolor='#000000')

    for i in range(0, BOARD_SIZE) :
        plt.axhline(y=i, color='black', linewidth=1)
        plt.axvline(x=i, color='black', linewidth=1)

    plt.gca().invert_yaxis()
    plt.title("Board")

    plt.draw()
    plt.pause(pause_time)

    if (save_image is True) :
        plt.savefig(save_image_name)
    
    return

# Draw win probability graph
def Draw_Win_Graph(win_prob_list,
                   figure_num=5,
                   pause_time=0.1,
                   save_image_name=None) :
    
    plt.figure(figure_num, clear=True)
    plt.plot(win_prob_list) # X : [0 ... N-1]

    #plt.draw()
    plt.pause(pause_time)

    if (save_image_name is not None) :
        plt.savefig(save_image_name)

    return

def Draw_Win_Spot(win_spot,
                  figure_num=4,
                  pause_time=0.1,
                  title=None) :

    plt.figure(figure_num, clear=True)

    for i in range(0, BOARD_SIZE) :
        plt.axhline(y=i, color='black', linewidth=1)
        plt.axvline(x=i, color='black', linewidth=1)

    # mark max value as red, else black
    max_val = MIN_VALUE

    for x in range(0, BOARD_SIZE) :
        for y in range(0, BOARD_SIZE) :
            if (win_spot[y][x] > max_val) :
                max_val = win_spot[y][x]
                max_y = y
                max_x = x

    for x in range(0, BOARD_SIZE) :
        for y in range(0, BOARD_SIZE) :
            if (y == max_y) and (x == max_x) :
                color = "#ff0000"
            else :
                color = "#000000"
    
            plt.scatter(x, y, s=win_spot[y][x] * 400, c=color)

    # plt.imshow(win_spot, vmin=1e-3)
    plt.gca().invert_yaxis()

    if (title is not None) :
        plt.title(title)

    plt.draw()
    plt.pause(pause_time)

    return        
            

# Record something into string
def Record(string,
           filename,
           append_mode=True) :
    
    if (append_mode is True) :
        file = open(filename, "a")
    else :
        file = open(filename, "w")

    file.write(str(string))
    file.close()

    return
