# -*- coding: utf-8 -*-
'''
    Created by hushiwei on 2019/6/24
    Desc : 
    Note : 
'''

import numpy as np

x = [1, 2]
state = [0.0, 0.0]

# weight
w_cell_state = np.array([[0.1, 0.1],
                         [0.3, 0.4]])

w_cell_input = np.array([0.5, 0.6])
b_cell = np.array([0.1, -0.1])

# output weight
w_output = np.array([[1.0],
                     [2.0]])
b_output = 0.1

for i in range(len(x)):
    before_activation = np.dot(state, w_cell_state) + x[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # output
    final_output = np.dot(state, w_output) + b_output
    print("before activation: ", before_activation)
    print('state: ', state)
    print('output: ', final_output)
