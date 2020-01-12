import model
import game_colosseum

BLACK = 1
WHITE = -1

# Create 2 model
black_model = model.RL_Model(print_summary=True)
white_model = model.RL_Model()

black_model.restore_model(restore_train_counts=-1)
white_model.restore_model(restore_train_counts=-1)

train_count = 0

while (True) :  
    # Train count increases
    train_count += 1

    # Prepare game
    black_model.set_team(BLACK)
    white_model.set_team(WHITE)

    # Play game
    winner = game_colosseum.Game_Collosseum(black_model, white_model, train_count)

    # Train
    black_model.train_func(winner)
    white_model.train_func(winner)

    # Save model
    black_model.save_model(train_count)
    white_model.train_func(train_count)
    
