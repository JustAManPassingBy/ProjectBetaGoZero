# This engine seems to cause speed down.


GAME_TYPE = "BADUK"

# Game state Hub
def get_game_state(GameEngine) :
    if (GAME_TYPE is "BADUK") :
        return GameEngine.show_4_latest_boards()
    else :
        print("get_game_state, unknown game type : " + str(GAME_TYPE))
        return None

# Get valid state hub
def get_valid_states(GameEngine) :
    if (GAME_TYPE is "BADUK") :
        return GameEngine.get_legal_moves(include_eyes=True)
    else :
        print("get_valid_states, unkwnon game type : " + str(GAME_TYPE))
        return None

def set_move(GameEngine, action) :
    if (GAME_TYPE is "BADUK") :
        return GameEngine.do_move(action)
    else :
        print("set_move, unknown game type : ", str(GAME_TYPE))
        return None