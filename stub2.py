# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import math
from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # Still need to take into account.
        self.gravity = 4


        # Discretize.
        self.velocity_bin = 20
        self.velocity_range = (-40, 40)

        self.tree_dist_bin = 20
        self.tree_dist_range = (-200, 500)

        self.top_dif_bin = 20
        self.top_dif_range = (-400, 400)

        # Q Table.
        self.dimensions = (self.velocity_bin, self.tree_dist_bin, self.top_dif_bin)
        self.Q = np.zeros(self.dimensions + (2,))

        self.epsilon = 0.5
        self.gamma = 1
        self.alpha = 0.001

        self.iteration = 0

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.gravity = 4
        self.iteration += 1


    def sortIntoBin(self, value, rge, bin):
        size = (rge[1] - rge[0])/bin
        return math.floor((value - rge[0])/(size))

    # Takes state dictionary and returns a tuple.
    def stateConversion(self, state):
        velocity = self.sortIntoBin(state['monkey']['vel'], self.velocity_range, self.velocity_bin)
        tree_dist = self.sortIntoBin(state['tree']['dist'], self.tree_dist_range, self.tree_dist_bin)

        heightDif = state['tree']['top'] - state['monkey']['top']
        top_dif = self.sortIntoBin(heightDif, self.top_dif_range, self.top_dif_bin)

        return (velocity, tree_dist, top_dif)


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # For the first iteration.
        if(self.last_state == None):
            new_action = 0
            self.last_state = state
            self.last_action = int(new_action)
            return self.last_action

        if(self.last_state and self.gravity < 0):
            self.gravity = state['monkey']['vel'] - self.last_state['monkey']['vel']
        
        if self.gravity == 4:
            if(state['monkey']['vel'] > 20):
                print("Heuristic")
                self.last_state = state
                self.last_action = 0
                return self.last_action

            if(state['monkey']['vel'] < -30):
                print("Heuristic")
                self.last_state = state
                self.last_action = 1
                return self.last_action

            if(state['tree']['dist'] < 400 and state['monkey']['top'] > 275):
                print("Heuristic")
                self.last_state = state
                self.last_action = 0
                return self.last_action

            if(state['tree']['dist'] < 400 and state['monkey']['top'] < 75):
                print("Heuristic")
                self.last_state = state
                self.last_action = 1
                return self.last_action

            if(state['tree']['dist'] < 400 and state['monkey']['bot'] -  state['tree']['bot'] < 75):
                print("Heuristic")
                self.last_state = state
                self.last_action = 1
                return self.last_action

            if(state['tree']['dist'] < 400 and state['tree']['top'] -  state['monkey']['top'] < 75):
                print("Heuristic")
                self.last_state = state
                self.last_action = 0
                return self.last_action
        else:
            if(state['monkey']['vel'] > 15):
                print("Heuristic")
                self.last_state = state
                self.last_action = 0
                return self.last_action

            if(state['monkey']['vel'] < -25):
                print("Heuristic")
                self.last_state = state
                self.last_action = 1
                return self.last_action

            if(state['tree']['dist'] < 250 and state['monkey']['top'] > 250):
                print("Heuristic")
                self.last_state = state
                self.last_action = 0
                return self.last_action

            if(state['tree']['dist'] < 300 and state['monkey']['top'] < 100):
                print("Heuristic")
                self.last_state = state
                self.last_action = 1
                return self.last_action

            if(state['tree']['dist'] < 300 and state['monkey']['bot'] -  state['tree']['bot'] < 100):
                print("Heuristic")
                self.last_state = state
                self.last_action = 1
                return self.last_action

            if(state['tree']['dist'] < 300 and state['tree']['top'] -  state['monkey']['top'] < 100):
                print("Heuristic")
                self.last_state = state
                self.last_action = 0
                return self.last_action
        

        # Do some Q-learning.
        s = self.stateConversion(state)
        Q_max = np.max(self.Q[s])


        ls = self.stateConversion(self.last_state)

        a = (self.last_action,)
        self.Q[ls + a] = self.Q[ls + a] - self.alpha*(self.Q[ls + a] - (self.last_reward + self.gamma*Q_max))



        # print("Velocity:", state['monkey']['vel'])
        # print("Distance to next tree trunk", state['tree']['dist'])
        # print("Diff between tops: ", state['tree']['top'] - state['monkey']['top'])
        # print("Diff between bottoms: ", state['monkey']['bot'] - state['tree']['bot'])

        # Choose action from epsilon-greedy policy.
        amax = np.argmax(self.Q[s])
        new_action = npr.choice([amax, 1-amax], p=[1.0 - self.epsilon, self.epsilon])
        if self.epsilon >= 0.01: 
            self.epsilon -= .0005
        print("Q Table")

        self.last_action = new_action
        self.last_state  = state
        #print(self.epsilon)
        return self.last_action

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
        print("Death")
        # Save score history.
        hist.append(swing.score)
        #print(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return




if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 1000, 80)

	# Save history.
	np.save('hist',np.array(hist))
