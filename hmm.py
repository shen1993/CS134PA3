"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
from two_way_dict import TwoWayDict
from collections import defaultdict
import numpy


class HMM(Classifier):
    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    model = property(get_model, set_model)

    def initialize_w_mode(self):
        self.word_no_pos = True

    def initialize_p_mode(self):
        self.pos_no_word = True

    def __init__(self):
        self.word_no_pos = False
        self.pos_no_word = False
        self.UNK = '<UNK>'
        self.labels = []
        self.features = []
        self.feature_2wdict = TwoWayDict()
        self.label_2wdict = TwoWayDict()
        self.start_prob_dict = {}
        self.transition_matrix = numpy.zeros((1, 1))
        self.emission_matrix = numpy.zeros((1, 1))
        self.transition_count_table = numpy.zeros((1, 1))
        self.feature_count_table = numpy.zeros((1, 1))

        numpy.set_printoptions(suppress=True)

    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function should update self.transtion_count_table
        and self.feature_count_table based on this new given instance
        
        Update self.transtion_count_table and self.feature_count_table based on
        the features. If feature unknown then apply self.UNK

        Returns None
        """
        for instance in instance_list:
            for label, feature in zip(instance.label, instance.data):
                l = self.label_2wdict[label]
                if feature in self.features:
                    f = self.feature_2wdict[feature]
                else:
                    f = self.feature_2wdict[self.UNK]
                self.feature_count_table[l, f] += 1.0
            for i, label in enumerate(instance.label):
                if i == 0:
                    self.start_prob_dict[label] += 1.0
                else:
                    temp = instance.label[i - 1]
                    l1 = self.label_2wdict[temp]
                    l2 = self.label_2wdict[label]
                    self.transition_count_table[l1, l2] += 1.0

    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Update codebooks from the given data to be consistent with
        the probability tables 

        Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate 
        of the appropriate parameters

        First initialize self.labels and self.features, and create tables and matrices.
        Then self._collect_counts(instance_list)
        Finally get matrices based on the counts

        Returns None
        """
        # for instance in instance_list:
        #     if self.word_no_pos:
        #         instance.data = instance.data1
        #     elif self.pos_no_word:
        #         instance.data = instance.data2

        self._update(instance_list)

        self._collect_counts(instance_list)

        # normalize start probability
        total = sum(self.start_prob_dict.values())
        for item in self.start_prob_dict:
            self.start_prob_dict[item] /= total

        # estimate the parameters from the count tables
        self.transition_matrix = self.transition_count_table / self.transition_count_table.sum(axis=0)
        self.emission_matrix = self.feature_count_table / self.feature_count_table.sum(axis=0)

        print("\n", self.transition_matrix)
        print(self.emission_matrix)
        print(self.transition_count_table)
        print(self.feature_count_table)
        print(self.start_prob_dict)

    def classify(self, instance):
        """Viterbi decoding algorithm

        Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix

        First get the last label based on trellis then go backwards using backtrace_pointers to get all labels.

        Returns a list of labels e.g. ['B','I','O','O','B']
        """
        trellis, backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        best_sequence = []
        end_label = numpy.zeros(len(self.labels))
        for label in self.labels:
            i = self.label_2wdict[label]
            end_label[i] = trellis[i, -1]

        temp_list = [numpy.argmax(end_label)]
        for i in range(len(instance.data) - 1, 0, -1):
            temp_list.append(backtrace_pointers[int(temp_list[-1]), i])
        for j in temp_list[::-1]:
            best_sequence.append(self.label_2wdict[j])
        return best_sequence

    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        trellis = self.dynamic_programming_on_trellis(instance, True)
        loglikelihood = 0.0
        return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        First initialize the start probability,
        then fill up the dynamic programming table and get backtrace pointers.

        Returns trellis filled up with the forward probabilities 
        and backtrace pointers for finding the best sequence
        """

        # Initialize trellis and backtrace pointers

        trellis = numpy.zeros((len(self.labels), len(instance.data)))
        backtrace_pointers = numpy.zeros((len(self.labels), len(instance.data)))

        # initialize the start probability for each label
        for label in self.labels:
            l = self.label_2wdict[label]
            start_transition = self.start_prob_dict[label]
            if instance.data[0] in self.features:
                f = self.feature_2wdict[instance.data[0]]
            else:
                f = self.feature_2wdict[self.UNK]
            emission = self.emission_matrix[l, f]
            trellis[l, 0] = start_transition * emission

        # Traverse through the trellis
        for i in range(1, len(instance.data)):
            for label in self.labels:
                alpha = trellis[:, i - 1]
                l = self.label_2wdict[label]
                transition = self.transition_matrix[:, l]
                if instance.data[i] in self.features:
                    f = self.feature_2wdict[instance.data[i]]
                else:
                    f = self.feature_2wdict[self.UNK]
                emission = self.emission_matrix[l, f]
                if run_forward_alg:
                    trellis[l, i] = sum(alpha * transition * emission)
                else:
                    trellis[l, i] = max(alpha * transition * emission)
                    backtrace_pointers[l, i] = numpy.argmax(alpha * transition * emission)
        if run_forward_alg:
            return trellis
        else:
            return trellis, backtrace_pointers

    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        """Baum-Welch algorithm for fitting HMM from unlabeled data

        The algorithm first initializes the model with the labeled data if given.
        The model is initialized randomly otherwise. Then it runs 
        Baum-Welch algorithm to enhance the model with more data.

        Add your docstring here explaining how you implement this function

        Returns None
        """
        if labeled_instance_list is not None:
            self.train(labeled_instance_list)
        else:
            # initialize the model randomly
            self._update(unlabeled_instance_list)
            self._collect_counts(unlabeled_instance_list)
            # normalize start probability
            total = sum(self.start_prob_dict.values())
            for item in self.start_prob_dict:
                self.start_prob_dict[item] /= total
            # estimate the parameters from the count tables
            self.transition_matrix = self.transition_count_table / self.transition_count_table.sum(axis=0)
            self.emission_matrix = self.feature_count_table / self.feature_count_table.sum(axis=0)

        old_likelihood = 0
        likelihood = 0
        while True:
            # E-Step
            for j, instance in enumerate(unlabeled_instance_list):
                if j % 1000 == 0:
                    print("Running..instance ", j)
                alpha_table, beta_table = self._run_forward_backward(instance)
                gamma_table = alpha_table * beta_table
                gamma_table /= gamma_table.sum(axis=0)
                # update the expected count tables based on alphas and betas
                for label in instance.label:
                    for i, feature in enumerate(instance.data):
                        l = self.label_2wdict[label]
                        if feature in self.features:
                            f = self.feature_2wdict[feature]
                        else:
                            f = self.feature_2wdict[self.UNK]
                        self.expected_transition_counts[l, :] = self.transition_count_table[l, :] * gamma_table[l, i]
                        self.expected_feature_counts[l, f] = self.feature_count_table[l, f] * gamma_table[l, i] * 1.0
                        # print(self.expected_feature_counts[l,f], self.feature_count_table[l, f], gamma_table[l, i])

                        # also combine the expected count with the observed counts from the labeled data
            # M-Step
            # re-estimate the parameters
            self.transition_count_table = self.expected_transition_counts
            self.feature_count_table = self.expected_feature_counts
            self.transition_matrix = self.transition_count_table / self.transition_count_table.sum(axis=0)
            self.emission_matrix = self.feature_count_table / self.feature_count_table.sum(axis=0)

            likelihood += 1
            if self._has_converged(old_likelihood, likelihood):
                break

    def _has_converged(self, old_likelihood, likelihood):
        """Determine whether the parameters have converged or not

        Returns True if the parameters have converged.    
        """
        if likelihood - old_likelihood > 10:
            return True

    def _run_forward_backward(self, instance):
        """Forward-backward algorithm for HMM using trellis
    
        Fill up the alpha and beta trellises (the same notation as 
        presented in the lecture and Martin and Jurafsky)
        You can reuse your forward algorithm here

        return a tuple of tables consisting of alpha and beta tables
        """
        # implement forward backward algorithm
        beta_table = numpy.zeros((len(self.labels), len(instance.data)))
        alpha_table = self.dynamic_programming_on_trellis(instance, True)

        for label in self.labels:
            l = self.label_2wdict[label]
            beta_table[l, len(instance.data) - 1] = 1.0
        for i in range(len(instance.data) - 1, 0, -1):
            for label in self.labels:
                beta = beta_table[:, i]
                l = self.label_2wdict[label]
                transition = self.transition_matrix[:, l]
                if instance.data[i] in self.features:
                    f = self.feature_2wdict[instance.data[i]]
                else:
                    f = self.feature_2wdict[self.UNK]
                emission = self.emission_matrix[l, f]
                beta_table[l, i - 1] = sum(beta * transition * emission)
        return alpha_table, beta_table

    def _update(self, instance_list):
        # initialize tables and matrices and other variables
        labels = []
        features = []
        for instance in instance_list:
            labels += instance.label
            features += instance.data
        labels = set(labels)
        features = set(features)
        features.add(self.UNK)
        self.labels = labels
        self.features = features

        self.label_2wdict = TwoWayDict()
        self.feature_2wdict = TwoWayDict()
        for i, label in enumerate(labels):
            self.label_2wdict[i] = label
        for i, feature in enumerate(features):
            self.feature_2wdict[i] = feature

        self.transition_matrix = numpy.zeros((len(labels), len(labels)))
        self.emission_matrix = numpy.zeros((len(labels), len(features)))
        self.transition_count_table = numpy.ones((len(labels), len(labels)))  # smoothing
        self.feature_count_table = numpy.ones((len(labels), len(features)))
        self.start_prob_dict = defaultdict(int)

        self.expected_transition_counts = numpy.zeros((len(labels), len(labels)))
        self.expected_feature_counts = numpy.zeros((len(labels), len(features)))
