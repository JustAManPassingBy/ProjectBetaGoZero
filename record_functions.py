import numpy as np
from matplotlib import pyplot as plt

from definitions import BOARD_SIZE, SQUARE_BOARD_SIZE, DEFAULT_GAME_COUNT

# Draw plot with Board
def Draw_Plot(board, 
              figure_num=3, 
              pause_time=0.1,
              save_image=False,
              save_image_name=None) :

    plt.figure(figure_num, clear=True)

    plt.imshow(board)

    for i in range(0, BOARD_SIZE) :
        plt.axhline(y=i, color='black', linewidth=1)
        plt.axvline(x=i, color='black', linewidth=1)

    plt.title("Board")
    plt.colorbar()  

    plt.draw()
    plt.pause(pause_time)

    if (save_image is True) :
        plt.savefig(save_image_name)
    
    return

# Draw win probability graph
def Draw_Win_Graph(win_prob_list,
                   figure_num=5,
                   save_image_name=None) :
    
    plt.figure(figure_num, clear=True)
    plt.plot(win_prob_list) # X : [0 ... N-1]

    plt.savefig(save_image_name)

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
