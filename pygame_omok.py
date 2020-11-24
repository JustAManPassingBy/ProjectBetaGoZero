import pygame
import model
from pygame.locals import *
from game import *
from main import *
from game_omok import OmokGame
from game_colosseum import *
import copy

from definitions import WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT, BG_COLOR, GRID_SIZE, BOARD_SIZE
from definitions import TRAIN_COUNTS_TO_SKIP_NONE_MOVE

fps = 60
fps_clock = pygame.time.Clock()

# colors
black = (10, 10, 10)
blue = (10, 10, 255)
green = (10, 255, 10)
white = (245, 245, 245)
red = (255, 10, 10)

board_bg = (222, 184, 135)

BOARD_INTERVAL = 20
STONE_SIZE = 10
STONE_ADD_RECT = 2

USER_BLACK = 1

class Screen() :
    def __init__(self, surface, Gamemode=None, GameEngine=None) :
        self.surface = surface
        self.Gamemode = Gamemode
        self.GameEngine = GameEngine
        self.pixel_coords = []
        self._set_pixel_coords()

        self.font = pygame.font.SysFont('Arial', 10)

    def make_text(self, font, text, color, bgcolor, top, left, position=0):
        surf = font.render(text, False, color, bgcolor)
        rect = surf.get_rect()
        if position:
            rect.center = (left, top)
        else:    
            rect.topleft = (left, top)
        self.surface.blit(surf, rect)
        return rect

    def _draw_board(self, board, last_action, last_action_color=None) :
        # Draw Background
        pygame.draw.rect(self.surface, board_bg, [0, 0, BOARD_WIDTH, BOARD_HEIGHT])

        # Transpose Board
        board = np.transpose(board)

        # Draw Line
        start_x = start_y = BOARD_INTERVAL
        end_x = BOARD_WIDTH - BOARD_INTERVAL
        end_y = BOARD_HEIGHT - BOARD_INTERVAL

        x_interval = int((end_x - start_x) / (BOARD_SIZE - 1))
        y_interval = int((end_y - start_y) / (BOARD_SIZE - 1))

        for i in range(BOARD_SIZE) :
            sx = start_x + i * x_interval
            sy = start_y + i * y_interval

            pygame.draw.line(self.surface, black, [sx, start_y], [sx, end_y], 3)
            pygame.draw.line(self.surface, black, [start_x, sy], [end_x, sy], 3)

        # Draw Stones
        if (last_action is not None) :
            (last_x, last_y) = last_action

            if (board[last_y][last_x] > 0) :
                black_first = 1
            elif (board[last_y][last_x] < 0) :
                black_first = 0
            else :
                black_first = -1
        else :
            black_first = -1
        

        for y in range(0, BOARD_SIZE) :
            spot_y = int(start_y + y_interval * y)
            for x in range(0, BOARD_SIZE) :
                spot_x = int(start_x + x_interval * x)

                # Mark Stones (last move, black, white)
                if (last_action is not None) and ((y == last_y) and (x == last_x)) :
                    pygame.draw.circle(self.surface, last_action_color, [spot_x, spot_y], STONE_SIZE)
                elif (board[y][x] > 0) :
                    pygame.draw.circle(self.surface, black, [spot_x, spot_y], STONE_SIZE)
                elif (board[y][x] < 0) :
                    pygame.draw.circle(self.surface, white, [spot_x, spot_y], STONE_SIZE)

                # Mark Number (For give predict informations)
                if (black_first is not -1) :
                    if (board[y][x] > 1) :
                        if (black_first is 1) :
                            number = (board[y][x] - 1) * 2 - 1
                        else :
                            number = (board[y][x] - 1) * 2
                    elif (board[y][x] < -1) :
                        if (black_first is 1) :
                            number = ((-board[y][x]) - 1) * 2
                        else :
                            number = (((-board[y][x]) - 1) * 2) - 1
                    else :
                        number = None

                    if (number is not None) :
                        number_string = str(number)

                        if (board[y][x] > 0) :
                            _ = self.make_text(self.font, number_string, white, None, spot_y - int(STONE_SIZE * 0.6), spot_x - int(STONE_SIZE * 0.6))
                        elif (board[y][x] < 0):
                            _ = self.make_text(self.font, number_string, black, None, spot_y - int(STONE_SIZE * 0.6), spot_x - int(STONE_SIZE * 0.6))

        pygame.display.update()

    # Create area of each stone's range (coord)
    def _set_pixel_coords(self) :
        start_x = start_y = BOARD_INTERVAL
        end_x = BOARD_WIDTH - BOARD_INTERVAL
        end_y = BOARD_HEIGHT - BOARD_INTERVAL

        x_interval = int((end_x - start_x) / (BOARD_SIZE - 1))
        y_interval = int((end_y - start_y) / (BOARD_SIZE - 1))
    
        for y in range(BOARD_SIZE) :
            spot_y = int(start_y + y_interval * y - STONE_SIZE / 2)

            for x in range(BOARD_SIZE) :
                spot_x = int(start_x + x_interval * x - STONE_SIZE / 2)

                self.pixel_coords.append((spot_x, spot_y, x, y))

                print(spot_x, spot_y, x, y)


    def check_board(self, board, pos) :
        # Transpose Board
        board = np.transpose(board)

        print(pos)
        for coord in self.pixel_coords :
            x, y, valx, valy = coord
            rect = pygame.Rect(x - STONE_ADD_RECT, y - STONE_ADD_RECT, STONE_SIZE + 2 * STONE_ADD_RECT, STONE_SIZE + 2 * STONE_ADD_RECT)

            if rect.collidepoint(pos) :
                if (board[valy][valx] == EMPTY) :
                    return (valx, valy)

        return None

    def check_board_without_click(self, board, pos) :
        # Transpose Board
        board = np.transpose(board)

        for coord in self.pixel_coords :
            x, y, valx, valy = coord
            rect = pygame.Rect(x - STONE_ADD_RECT, y - STONE_ADD_RECT, STONE_SIZE + 2 * STONE_ADD_RECT, STONE_SIZE + 2 * STONE_ADD_RECT)

            if rect.collidepoint(pos) :
                return (valx, valy)

        return None

class Menu(object):
    def __init__(self, surface):
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.surface = surface
        self.draw_menu()

    def draw_menu(self):
        top, left = WINDOW_HEIGHT - 30, WINDOW_WIDTH - 150
        self.new_rect = self.make_text(self.font, 'New Game', blue, None, top - 30, left)
        self.reset_rect = self.make_text(self.font, 'Reset', blue, None, top - 60, left)
        self.end_rect = self.make_text(self.font, 'end', blue, None, top - 90, left)

    def make_text(self, font, text, color, bgcolor, top, left, position = 0):
        surf = font.render(text, False, color, bgcolor)
        rect = surf.get_rect()
        if position:
            rect.center = (left, top)
        else:    
            rect.topleft = (left, top)
        self.surface.blit(surf, rect)
        return rect

    def check_rect(self, pos):
        if self.new_rect.collidepoint(pos):
            return 1 # Start new game
        elif self.reset_rect.collidepoint(pos):
            return 2 # Reset Game
        elif self.end_rect.collidepoint(pos) :
            return 3 # End game

        return 0

# Is turn white is -1(False) or 1(True), not 0(False) or 1(True)
def create_history_board(board, move_history, is_turn_white) :
    new_board = copy.deepcopy(board)

    white_count = -2
    black_count = 2

    for i in range(0, len(move_history), 1) :
        if (move_history[i] is None) :
            continue
        (x, y) = move_history[i]

        if (is_turn_white == 1) :
            new_board[x][y] = white_count
            white_count -= 1
        else :
            new_board[x][y] = black_count
            black_count += 1

        is_turn_white *= -1

    return new_board

def run_game(surface, board, menu) :
    TS_dummy = Train_Sample()

    _ , black_model = main_init_model(TS_dummy)

    white_model = black_model

    gameengine = GameState()
    gameboard = gameengine.show_result()
    
    board._draw_board(gameboard, (-1, -1), None)
    
    # If Your turn is black
    if (USER_BLACK is 1) :
        last_pc_move = (-10, -10) # Initial last pc move = None
        last_pc_gameboard = None

    # If Your turn is white
    else :
        game_end, winner, gameengine, action, best_actions = Game_Collosseum_Onemove(gameengine, black_model, white_model, 
                                                                                     TRAIN_COUNTS_TO_SKIP_NONE_MOVE + 1, 0)
 
        print(best_actions)

        gameboard = gameengine.show_result()
        board._draw_board(gameboard, action, green)

        last_pc_move = action
        last_pc_gameboard = create_history_board(gameboard, best_actions, USER_BLACK) # If user is black, AI Turn is White

    last_pc_printed = False

    while True :
        for event in pygame.event.get() :
            if event.type == QUIT :
                board.terminate()
                menu.terminate()
            elif event.type == MOUSEMOTION :
                check_board_ret = board.check_board_without_click(gameboard, event.pos)

                if (last_pc_move == check_board_ret) and (last_pc_printed is False) :
                    board._draw_board(last_pc_gameboard, action, green)
                    last_pc_printed = True

                elif (last_pc_printed is True) :
                    board._draw_board(gameboard, action, green)
                    last_pc_printed = False
            
            elif event.type == MOUSEBUTTONUP :
                check_board_ret = board.check_board(gameboard, event.pos)
                if check_board_ret is not None :
                    # Your turn
                    action = (x, y) = check_board_ret 

                    gameengine.do_move(action)
                    gameboard = gameengine.show_result()

                    board._draw_board(gameboard, action, red)

                    # AI's Turn
                    game_end, winner, gameengine, action, best_actions = Game_Collosseum_Onemove(gameengine, black_model, white_model, 0, 0)

                    print(best_actions)

                    gameboard = gameengine.show_result()
                    board._draw_board(gameboard, action, green)

                    last_pc_move = action
                    last_pc_gameboard = create_history_board(gameboard, best_actions, USER_BLACK)
                else :
                    check_menu_ret = menu.check_rect(event.pos)

                    # Start New Game / Reset Game
                    if (check_menu_ret == 1) or (check_menu_ret == 2) :
                        del gameengine

                        gameengine = GameState()
                        gameboard = gameengine.show_result()
                        board._draw_board(gameboard, (-1, -1))

                    # End Game
                    elif (check_menu_ret == 3) :
                        return

        pygame.display.update()
        fps_clock.tick(fps)

if __name__ == "__main__" :
    pygame.init()

    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Play Game!")
    surface.fill(BG_COLOR)

    board_screen = Screen(surface)
    menu_screen = Menu(surface)

    # Game play
    run_game(surface, board_screen, menu_screen)




