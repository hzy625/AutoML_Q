import os
import pandas as pd
import q_learner
import traceback
import time

class NAS():
    def __init__(self,
                 list_path,
                 state_space_parameters,
                 hyper_parameters,
                 epsilon=None,
                 number_models=None):

        self.replay_columns = ['net',                   #Net String
                               'accuracy_best_val',
                               'epsilon',
                               'iteration']         


        self.list_path = list_path

        self.replay_dictionary_path = os.path.join(list_path, 'replay_database.csv')
        self.replay_dictionary, self.q_training_step = self.load_replay()

        self.schedule_or_single = False if epsilon else True
        if self.schedule_or_single:
            self.epsilon = state_space_parameters.epsilon_schedule[0][0]
            self.number_models = state_space_parameters.epsilon_schedule[0][1]
        else:
            self.epsilon = epsilon
            self.number_models = number_models if number_models else 10000000000
        self.state_space_parameters = state_space_parameters
        self.hyper_parameters = hyper_parameters

        self.number_q_updates_per_train = 100

        self.list_path = list_path
        self.qlearner = self.load_qlearner()
        self.check_reached_limit()


    def load_replay(self):
        if os.path.isfile(self.replay_dictionary_path):
            print ('Found replay dictionary')
            replay_dic = pd.read_csv(self.replay_dictionary_path)
            q_training_step = max(replay_dic.iteration)
        else:
            replay_dic = pd.DataFrame(columns=self.replay_columns)
            q_training_step = 0
        return replay_dic, q_training_step

    def load_qlearner(self):
        # Load previous q_values
        if os.path.isfile(os.path.join(self.list_path, 'q_values.csv')):
            print ('Found q values')
            qstore = q_learner.QValues()
            qstore.load_q_values(os.path.join(self.list_path, 'q_values.csv'))
        else:
            qstore = None


        ql = q_learner.QLearner(self.state_space_parameters,
                                    self.epsilon,
                                    qstore=qstore,
                                    replay_dictionary=self.replay_dictionary)

        return ql

    def filter_replay_for_first_run(self, replay):
        ''' Order replay by iteration, then remove duplicate nets keeping the first'''
        temp = replay.sort_values(['iteration']).reset_index(drop=True).copy()
        return temp.drop_duplicates(['net'])

    def number_trained_unique(self, epsilon=None):
        '''Epsilon defaults to the minimum'''
        replay_unique = self.filter_replay_for_first_run(self.replay_dictionary)
        eps = epsilon if epsilon else min(replay_unique.epsilon.values)
        replay_unique = replay_unique[replay_unique.epsilon == eps]
        return len(replay_unique)

    def check_reached_limit(self):
        ''' Returns True if the experiment is complete
        '''
        if len(self.replay_dictionary):
            completed_current = self.number_trained_unique(self.epsilon) >= self.number_models

            if completed_current:
                if self.schedule_or_single:
                    # Loop through epsilon schedule, If we find an epsilon that isn't trained, start using that.
                    completed_experiment = True
                    for epsilon, num_models in self.state_space_parameters.epsilon_schedule:
                        if self.number_trained_unique(epsilon) < num_models:
                            self.epsilon = epsilon
                            self.number_models = num_models
                            self.qlearner = self.load_qlearner()
                            completed_experiment = False

                            break

                else:
                    completed_experiment = True

                return completed_experiment

            else:
                return False

    def generate_new_netork(self):
        try:
            (net,
             acc_best_val) = self.qlearner.generate_net()

            # We have already trained this net
            if net in self.replay_dictionary.net.values:
                self.q_training_step += 1
                self.incorporate_trained_net(net,
                                             acc_best_val,
                                             self.epsilon)
                return self.generate_new_netork()
            else:
                self.q_training_step += 1
                return net, self.q_training_step

        except Exception:
            print (traceback.print_exc())

    def incorporate_trained_net(self,
                                net_string,
                                acc_best_val,                         
                                epsilon,
                                iteration):

        try:
            # If we sampled the same net many times, we should add them each into the replay database
            self.replay_dictionary = pd.concat([self.replay_dictionary, pd.DataFrame({'net':[net_string],
                                                                                          'accuracy_best_val':[acc_best_val],
                                                                                          'epsilon': [epsilon],
                                                                                          'iteration' : [iteration]})])
            self.replay_dictionary.to_csv(self.replay_dictionary_path, index=False, columns=self.replay_columns)

            self.qlearner.update_replay_database(self.replay_dictionary)
            self.qlearner.sample_replay_for_update()
            self.qlearner.save_q(self.list_path)
            print ('Incorporated net, acc: %f, net: %s' % (acc_best_val, net_string))
        except Exception:
            print (traceback.print_exc())

            
            