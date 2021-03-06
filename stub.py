# Imports.
import numpy as np
import numpy.random as npr
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import copy

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None

        self.epsilon = None
        self.gamma = 1
        self.eta = .1
        self.epsilon = 1

        self.model = Sequential()
        self.model.add(Dense(1, init='lecun_uniform', input_shape=(7,)))
        self.model.add(Activation('sigmoid'))
        #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

        # self.model.add(Dense(50, init='lecun_uniform'))
        # self.model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        #Predict the value of the state
        #self.model.add(Dense(1, init='lecun_uniform'))
        self.model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)

    def reset(self):
        self.gravity = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def getGravity(self):
        return self.gravity

    #Flatten the state
    def __get_statevec(self, state):
        vec = [self.gravity, state['tree']['bot'], state['tree']['top'],state['tree']['dist'], state['monkey']['vel'],state['monkey']['bot'], state['monkey']['top']]
        return np.array(vec).reshape((1, len(vec)))


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.


        #We've done one iteration with the current game but have yet to update gravity
        if self.last_state and self.gravity < 0:
            self.gravity = np.abs(state['monkey']['vel'] - self.last_state['monkey']['vel']) #Velocity is in the y direction

        #We are treating the last state as s and the current state as s', in the eyes of our update
        if self.last_state:
            #Q(s', a') values for the 2 actions

            swing = copy.deepcopy(state)


            x_vel = state['tree']['dist'] - self.last_state['tree']['dist'] #Horizontal speed doesn't change
            y_vel = state['monkey']['bot'] - self.last_state['monkey']['bot']

            #This game doesnt obey basic fucking physics
            swing['monkey']['top'] -= y_vel
            swing['monkey']['bot'] += y_vel 
            swing['monkey']['vel'] -= self.gravity

            swing['tree']['top'] -= y_vel
            swing['tree']['bot'] += y_vel 
            swing['tree']['dist'] -= x_vel

            #Jumping
            impulse = 15 #The impulse is a poisson R.V with parameter 15
            jump = copy.deepcopy(state)
            jump['monkey']['top'] -= impulse #The impulse becomes the velocity
            jump['monkey']['bot'] += impulse
            jump['monkey']['vel'] = impulse - self.gravity

            jump['tree']['top'] -= impulse
            jump['tree']['bot'] += impulse
            jump['tree']['dist'] -= x_vel


            qval_swing = self.model.predict(self.__get_statevec(swing), batch_size=1)
            qval_jump = self.model.predict(self.__get_statevec(jump), batch_size=1)


            #Get max_Q(s',a'), we could change this to SARSA
            maxQ = max(qval_swing, qval_jump)

            #Observed Reward: r(s,a)
            reward = self.last_reward

            #Updating our w parameters based on Q(s,a)
            y = reward + (self.gamma * maxQ)


            self.model.fit(self.__get_statevec(state), y, batch_size=1, nb_epoch=1, verbose=1)


            #Choose an epsilon greedy policy for the current state 
            if self.epsilon > 0.1:
                self.epsilon -= .001


            if (npr.random() < self.epsilon): #choose random action
                action = np.random.randint(0,2)
            else: #choose best action from Q(s',a') values
                action = (np.argmax([qval_swing, qval_jump]))

        else:
            action = 0 #So that we can infer gravity, I'm open to better solutions to this (which is to just let the monkey swing in the first step)

        self.last_state = state
        self.last_action = action
        print self.last_action
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, score_hist, gravity_hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        score_hist.append(swing.score)
        gravity_hist.append(learner.getGravity())

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    score_hist = []
    gravity_hist = []


    # Run games. 
    run_games(agent, score_hist, gravity_hist, 1000, 1)

    # Save history. 
    np.save('score_hist',np.array(score_hist))
    np.save('gravity_hist', np.array(gravity_hist))