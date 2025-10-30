import random
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent():

    def __init__(self, state_size, hidden_layers_sizes, action_size):

        # Sizes of the different inputs and outputs of the network
        self.state_size = state_size
        self.hidden_layer_sizes = hidden_layers_sizes
        self.action_size = action_size

        # Memory
        self.memory_size = 100000
        self.memory = []

        # The size of each batch
        self.batch_size = 64

        # Hyperparameters
        self.alpha = 0.001 # learning rate (how fast the network updates its weights)
        self.gamma = 0.99  # discount factor (high means long-term vision, low means short-term vision)
        self.epsilon = 1.0 # exploration factor (start : exploration, end : exploitation)
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.model = self.build_model()
    
    def build_model(self):
        '''
        Build the initial model
        This model is composed of :
        - 1 input layer (state_size -> hidden_layer_sizes[0])
        - 1 to n hidden layers (hidden_layer_sizes[i] -> hidden_layer_sizes[i+1])
        - 1 output layer (hidden_layer_sizes[n] -> action_size)
        The loss function is mean squared error and the optimizer is Adam
        '''
        model = Sequential()

        model.add(Dense(self.hidden_layer_sizes[0], input_dim=self.state_size, activation='relu'))
        
        for n in self.hidden_layer_sizes:
            model.add(Dense(n, activation='relu'))
        
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha))

        return model
    
    def remember(self, exp):
        '''
        Save the experience in the memory
        '''
        self.memory.append(exp)
        if len(self.memory) > self.memory_size:
            del self.memory[0]
    
    def act(self, state):
        '''
        Chose an action depending on epsilon :
        - epsion chance of taking a random action
        - (1-epsilon) chance of taking the best action in the current model
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_ = np.array(state).reshape(1, self.state_size)
        action_values = self.model.predict(state_, verbose=0)
        return np.argmax(action_values[0])
    
    def sample(self, batch_size):
        '''
        Take a random batch of size batch_size in the memory
        '''
        samples = random.sample(self.memory, batch_size)
        obs, r, a, next_obs, done = zip(*samples)
        return list(obs), list(r), list(a), list(next_obs), list(done)

    def learn(self):
        '''
        Learn from the previous experiences
        '''

        # Take a random batch
        state, r, a, next_state, done = self.sample(self.batch_size)
    
        # Reshaping
        state_array = np.array(state).reshape(self.batch_size, self.state_size)
        next_state_array = np.array(next_state).reshape(self.batch_size, self.state_size)
        
        # Calculating Q-values for current_states and next_states 
        target_pred = self.model.predict(state_array, verbose=0)
        next_pred = self.model.predict(next_state_array, verbose=0)
        
        # Using Bellman equation :
        # r[i] + self.gamma * np.amax(next_pred[i])   means "immediate reward + best reward in the next state"
        # pred_target[i][a[i]] = ...                  means "now the value of this action describes how good is the action considering the future"
        for i in range(self.batch_size):
            if not done[i]:
                target_pred[i][a[i]] = r[i] + self.gamma * np.amax(next_pred[i])
            else:
                target_pred[i][a[i]] = r[i]
        
        # Adjusts weights in the network according to the predictions
        self.model.fit(state_array, target_pred, epochs=1, verbose=0)
    
    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = load_model(filename)
