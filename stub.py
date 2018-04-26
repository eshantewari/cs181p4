# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = None
        self.gamma = 1
        self.eta = .1
        self.epsilon = .5
        self.w = np.random.random(6)

    def reset(self):
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



        if self.last_action:
            q_swing_curr = state #How do we estimate Q(s',a';w)? This variable is Q(curr_state, swing; w), which we need to calculate to update w
            q_jump_curr = state #How do we estimate Q(s',a';w)? This variable is Q(curr_state, jump; w), which we need to calculate to update w

            dl_dw = (self.__get_q(self.last_state) - (self.last_reward + self.gamma * max(self.__get_q(q_swing_curr), self.__get_q(q_jump_curr)))) * self.w
            self.w = np.subtract(self.w, self.eta * dl_dw)

        if np.random.random() < self.epsilon:
            self.last_action = 1
        else:
            self.last_action = 0

        self.last_state  = state

        return self.last_action



    def __get_q(self, state):
        return np.dot(self.w, self.__get_statevec(state))

    #Flatten the state
    def __get_statevec(self, state):
        return [state['tree']['bot'], state['tree']['top'],state['tree']['dist'], state['monkey']['vel'],state['monkey']['bot'], state['monkey']['top']]

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


