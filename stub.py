# Imports.
import numpy as np
import numpy.random as npr
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
        self.epsilon = .5
        self.w = np.random.random(8) #6 state variables + gravity + bias

    def reset(self):
        self.gravity = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        print(self.w)
        if self.last_state and not self.gravity:
            self.gravity = np.abs(state['monkey']['vel'] - self.last_state['monkey']['vel'])


        if self.last_action >= 0:
            #Swinging, so going down with gravity
            q_swing_curr = copy.deepcopy(state)


            x_vel = state['tree']['dist'] - self.last_state['tree']['dist'] #Horizontal speed doesn't change
            y_vel = state['monkey']['bot'] - self.last_state['monkey']['bot']

            #This game doesnt obey basic fucking physics
            q_swing_curr['monkey']['top'] -= y_vel
            q_swing_curr['monkey']['bot'] += y_vel 
            q_swing_curr['monkey']['vel'] -= self.gravity

            q_swing_curr['tree']['top'] -= y_vel
            q_swing_curr['tree']['bot'] += y_vel 
            q_swing_curr['tree']['dist'] -= x_vel

            #Jumping
            impulse = 15 #The impulse is a poisson R.V with parameter 15
            q_jump_curr = copy.deepcopy(state)
            q_swing_curr['monkey']['top'] -= impulse #The impulse becomes the velocity
            q_swing_curr['monkey']['bot'] += impulse
            q_swing_curr['monkey']['vel'] = impulse - self.gravity

            q_swing_curr['tree']['top'] -= impulse
            q_swing_curr['tree']['bot'] += impulse
            q_swing_curr['tree']['dist'] -= x_vel



            dl_dw = (self.__get_q(self.last_state) - (self.last_reward + self.gamma * max(self.__get_q(q_swing_curr), self.__get_q(q_jump_curr)))) * self.w
            self.w = np.subtract(self.w, self.eta * dl_dw)
        


            if np.random.random() < self.epsilon:
                self.last_action = 1
            else:
                self.last_action = 0

        #We're on our difst action, which we want to be 0 to infer gravity        
        else:
            self.last_action = 0


        self.last_state  = state

        return self.last_action



    def __get_q(self, state):
        return np.dot(self.w, self.__get_statevec(state))

    #Flatten the state
    def __get_statevec(self, state):
        return [1, self.gravity, state['tree']['bot'], state['tree']['top'],state['tree']['dist'], state['monkey']['vel'],state['monkey']['bot'], state['monkey']['top']]

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
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
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 20, 10)

	# Save history. 
	np.save('hist',np.array(hist))


