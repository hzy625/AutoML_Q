import math
import numpy as np
from operator import itemgetter
#import cnn 
import lstm


# Action restrictions we have:  
# (1) we allow the agent to terminate a path at any point, i.e. it may choose a termination state from any non-termination state.  
# (2) we only allow transitions for a state with layer depth i to a state with layer depth i + 1 to ensure no loops in the graph.  
# (3) Any state at the maximum layer depth may only transition to a termination layer.  
# (4) we limit the number of fully connected (FC) layers in the whole network to be at maximum two to keep the set of learnable parameters reasonally small.   
# (5) Furthermore, even if (4) is satisfied, a state s of type FC with number of neurons d may only transition to another state s' of type FC with number of neurons d' <= d.  
# (6) no consecutive dropout layers.


class State:
    def __init__(self,
                 layer_type=None,        # String -- LSTM, dropout, fc, softmax
                 layer_depth=None,       # Current depth of network
#                  filter_depth=None,      # Used for conv, 0 when not conv
#                  filter_size=None,       # Used for conv and pool, 0 otherwise
#                  stride=None,            # Used for conv and pool, 0 otherwise
#                  image_size=None,        # Used for any layer that maintains square input (conv and pool), 0 otherwise
                 units=None,              # Used for LSTM, 0 when not LSTM
                 activation=None,         # Used for LSTM, fc, and equal to None otherwise
                 proba=None,              # Used for dropout, 0 otherwise
                 fc_depth=None,           # New added: Used for fc -- current number of fc layers used in network
                 fc_size=None,           # Used for fc and softmax -- number of neurons in layer, 0 otherwise
                 terminate=None,
                 state_list=None):       # can be constructed from a list instead, list takes precedent
        if not state_list: # if state_list is empty
            self.layer_type = layer_type
            self.layer_depth = layer_depth
#             self.filter_depth = filter_depth
#             self.filter_size = filter_size
#             self.stride = stride
#             self.image_size = image_size
            self.units = units
            self.activation = activation
            self.proba = proba
            self.fc_depth = fc_depth
            self.fc_size = fc_size
            self.terminate = terminate
        else: #if not empty
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
#             self.filter_depth = state_list[2]
#             self.filter_size = state_list[3]
#             self.stride = state_list[4]
#             self.image_size = state_list[5]
            self.units = state_list[2]
            self.activation = state_list[3]
            self.proba = state_list[4]  
            self.fc_depth = state_list[5]   
            self.fc_size = state_list[6]   
            self.terminate = state_list[7] # index adjusted after commented the 4 lines above
        
    def as_tuple(self):
        return (self.layer_type, 
                self.layer_depth, 
#                 self.filter_depth, 
#                 self.filter_size, 
#                 self.stride, 
#                 self.image_size,
                self.units,
                self.activation,
                self.proba,
                self.fc_depth,
                self.fc_size,
                self.terminate)
    
    def as_list(self):
        return list(self.as_tuple())
    
    def copy(self):
        return State(self.layer_type, 
                     self.layer_depth, 
#                      self.filter_depth, 
#                      self.filter_size, 
#                      self.stride, 
#                      self.image_size,
                     self.units,
                     self.activation,
                     self.proba,
                     self.fc_depth,
                     self.fc_size,
                     self.terminate)

class StateEnumerator:
    '''Class that deals with:
            Enumerating States (defining their possible transitions)

    '''
    def __init__(self, state_space_parameters):
        # Limits
        self.ssp = state_space_parameters
        self.layer_limit = state_space_parameters.layer_limit

        self.output_states = state_space_parameters.output_states 

    def enumerate_state(self, state, q_values):
        '''Defines all state transitions, populates q_values where actions are valid

        Legal Transitions:
           LSTM         -> LSTM, dropout        (IF state.layer_depth < layer_limit)
           LSTM         -> fc                (If state.layer_depth < layer_limit)
           LSTM         -> softmax              (Always)

           dropout      -> LSTM                 (If state.layer_depth < layer_limit)
           dropout      -> fc                (If state.layer_depth < layer_limit)
           dropout      -> softmax              (Always)

           fc        -> fc                (If state.layer_depth < layer_limit AND state.fc_depth < self.ssp.max_fc) # modified
           fc        -> dropout        (IF state.layer_depth < layer_limit)
           fc        -> softmax              (Always)

        Updates: q_values and returns q_values
        '''
        actions = []
        
        if state.terminate == 0:
            
            # First, add a terminate state (since the agent can choose to terminate from any non-termination state)
            # If we are at the layer limit, the only action left is to go to softmax
            actions += [State(layer_type=state.layer_type,
                                layer_depth=state.layer_depth + 1,
#                                 filter_depth=state.filter_depth,
#                                 filter_size=state.filter_size,
#                                 stride=state.stride,
#                                 image_size=state.image_size,
                                units=state.units,
                                activation=state.activation,
                                proba=state.proba,
                                fc_depth=state.fc_depth,
                                fc_size=state.fc_size,
                                terminate=1)]
            
            if state.layer_depth < self.layer_limit:

                # # Conv states -- iterate through all possible depths, filter sizes, and strides
                # if (state.layer_type in ['start', 'conv', 'pool']):        
                #     for depth in self.ssp.possible_conv_depths:
                #         for filt in self._possible_conv_sizes(state.image_size):
                #             actions += [State(layer_type='conv',
                #                                 layer_depth=state.layer_depth + 1,
                #                                 filter_depth=depth,
                #                                 filter_size=filt,
                #                                 stride=1,
                #                                 image_size=state.image_size if self.ssp.conv_padding == 'SAME' \
                #                                                             else self._calc_new_image_size(state.image_size, filt, 1),
                #                                 fc_size=0,
                #                                 terminate=0)]

                # (modified as below) 
                # lstm states -- iterate through all possible units (number of neurons on the layer), activation
                if (state.layer_type in ['start', 'LSTM', 'dropout']): # these layers can all go to lstm layer next
                    for unit in self.ssp.possible_units:
                        for activ_func in self.ssp.possible_actvf:
                            actions += [State(layer_type='LSTM',
                                                layer_depth=state.layer_depth + 1,
#                                                 filter_depth=depth,
#                                                 filter_size=filt,
#                                                 stride=1,
#                                                 image_size=state.image_size if self.ssp.conv_padding == 'SAME' \
#                                                                             else self._calc_new_image_size(state.image_size, filt, 1),
                                                units = unit,
                                                activation = activ_func,
                                                proba = 0,
                                                fc_depth = state.fc_depth,
                                                fc_size=0,
                                                terminate=0)]

                # Global Average Pooling States
#                 if (state.layer_type in ['start', 'conv', 'pool']):
#                     actions += [State(layer_type='gap',
#                                         layer_depth=state.layer_depth+1,
#                                         filter_depth=0,
#                                         filter_size=0,
#                                         stride=0,
#                                         image_size=1,
#                                         fc_size=0,
#                                         terminate=0)]

                # # pool states -- iterate through all possible filter sizes and strides
                # if (state.layer_type in ['conv'] or 
                #     (state.layer_type == 'pool' and self.ssp.allow_consecutive_pooling) or
                #     (state.layer_type == 'start' and self.ssp.allow_initial_pooling)): 
                #     for filt in self._possible_pool_sizes(state.image_size):
                #         for stride in self._possible_pool_strides(filt):
                #             actions += [State(layer_type='pool',
                #                                 layer_depth=state.layer_depth + 1,
                #                                 filter_depth=0,
                #                                 filter_size=filt,
                #                                 stride=stride,
                #                                 image_size=self._calc_new_image_size(state.image_size, filt, stride),
                #                                 fc_size=0,
                #                                 terminate=0)]
# I commented the blocks above out because we don't need pooling for LSTM here; thus we don't need a gap or pool layer for now.

                # dropout states -- iterate through all possible dropout rates (modified the original comment)
                if (state.layer_type in ['LSTM', 'fc'] or (state.layer_type == 'dropout' and self.ssp.allow_consecutive_dropout)): 
                    #or(state.layer_type == 'start' and self.ssp.allow_initial_pooling)): 
                    #commented the line above out since we don't do initial dropout at the start?
                    #do we need a layer type called 'start'?
                    for rate in self.ssp.possible_proba:
                        actions += [State(layer_type='dropout',
                                            layer_depth=state.layer_depth + 1,
#                                                 filter_depth=0,
#                                                 filter_size=filt,
#                                                 stride=stride,
#                                                 image_size=self._calc_new_image_size(state.image_size, filt, stride),
                                            units = 0,
                                            activation = None,
                                            proba = rate,
                                            fc_depth = state.fc_depth,
                                            fc_size=0,
                                            terminate=0)]

                # FC States -- iterate through all possible fc sizes
#                 if (self.ssp.allow_fully_connected(state.image_size)
#                     and state.layer_type in ['start', 'conv', 'pool']):

#                     for fc_size in self._possible_fc_size(state):
#                         actions += [State(layer_type='fc',
#                                             layer_depth=state.layer_depth + 1,
#                                             filter_depth=0,
#                                             filter_size=0,
#                                             stride=0,
#                                             image_size=0,
#                                             fc_size=fc_size,
#                                             terminate=0)]
# I commented the block above out because for LSTM, we have no restriction on transition to fc layer based on 
# current layer information except a restriction of max_fc and a restriction of units for consecutive fc layers 
# that we can have in a network.
# Instead, the new code is as below:

                # FC States -- iterate through all possible fc sizes
                if state.layer_type in ['start','fc','dropout','LSTM'] and state.fc_depth < self.ssp.max_fc - 1: #modified
                    for fc_size in self._possible_fc_size(state):
                        for activ_func in self.ssp.possible_actvf:
                            actions += [State(layer_type='fc',
                                                layer_depth=state.layer_depth + 1,
                                                fc_depth=state.fc_depth + 1,
                                                fc_size=fc_size,
                                                activation = activ_func,
    #                                             filter_size=0,
    #                                             stride=0,
    #                                             image_size=0,
                                                units = 0,
                                                proba = 0,
                                                terminate=0)]

        # Add states to transition and q_value dictionary (modified from original)
        q_values[state.as_tuple()] = {'actions': [to_state.as_tuple() for to_state in actions],
                                      'utilities': [self.ssp.init_utility for i in range(len(actions))]}
        return q_values        

    def transition_to_action(self, start_state, to_state): #modified as needed
        action = to_state.copy()
#         if to_state.layer_type not in ['fc', 'gap']:
#             action.image_size = start_state.image_size
        return action

    def state_action_transition(self, start_state, action):
        ''' start_state: Should be the actual start_state, not a bucketed state
            action: valid action

            returns: next state, not bucketed
        '''
#         if action.layer_type == 'pool' or \
#             (action.layer_type == 'conv' and self.ssp.conv_padding == 'VALID'):
#             new_image_size = self._calc_new_image_size(start_state.image_size, action.filter_size, action.stride)
#         else:
#             new_image_size = start_state.image_size

        to_state = action.copy()
#         to_state.image_size = new_image_size
        return to_state
                    
    def _possible_fc_size(self, state):
        '''Return a list of possible FC sizes given the current state'''
        if state.layer_type=='fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes

                    
# I commented out the following functions because they are not used in lstm case here
#     def bucket_state_tuple(self, state):
#         bucketed_state = State(state_list=state).copy()
#         bucketed_state.image_size = self.ssp.image_size_bucket(bucketed_state.image_size)
#         return bucketed_state.as_tuple()

#     def bucket_state(self, state):
#         bucketed_state = state.copy()
#         bucketed_state.image_size = self.ssp.image_size_bucket(bucketed_state.image_size)
#         return bucketed_state

#     def _calc_new_image_size(self, image_size, filter_size, stride):
#         '''Returns new image size given previous image size and filter parameters'''
#         new_size = int(math.ceil(float(image_size - filter_size + 1) / float(stride)))
#         return new_size

#     def _possible_conv_sizes(self, image_size):
#         return [conv for conv in self.ssp.possible_conv_sizes if conv < image_size]

#     def _possible_pool_sizes(self, image_size):
#         return [pool for pool in self.ssp.possible_pool_sizes if pool < image_size]

#     def _possible_pool_strides(self, filter_size):
#         return [stride for stride in self.ssp.possible_pool_strides if stride <= filter_size]


