from __future__ import division;
import graphlab;
import math;
from math import sqrt;
import json;
import numpy as np;

import matplotlib.pyplot as plt
# %matplotlib inline


import string;

products = graphlab.SFrame('../data/amazon_baby_subset.gl/');

with open('../important_words.json', 'r') as f:  # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)


products['review_clean'] = products['review'].apply(remove_punctuation)

for word in important_words:
    products[word] = products['review_clean'].apply(lambda s: s.split().count(word));

train_data, validation_data = products.random_split(.8, seed=2);

print 'Training set   : %d data points' % len(train_data);
print 'Validation set : %d data points' % len(validation_data);


def get_numpy_data(data_sframe, features, label):
    data_sframe['intercept'] = 1;
    features = ['intercept'] + features;
    features_sframe = data_sframe[features];
    feature_matrix = features_sframe.to_numpy();
    label_sarray = data_sframe[label];
    label_array = label_sarray.to_numpy();
    return (feature_matrix, label_array);


feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment')


def predict_probability(feature_matrix, coefficients):
    '''
    produces probablistic estimate for P(y_i = +1 | x_i, w).
    estimate ranges between 0 and 1.
    '''
    # Take dot product of feature_matrix and coefficients
    # YOUR CODE HERE
    dot_product = np.dot(feature_matrix, coefficients);

    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    predictions = 1 / (1 + np.exp(-1 * dot_product));

    # return predictions
    return predictions;


def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant):
    # type: (errors, feature, coefficient, l2_penalty, feature_is_constant) -> np
    # Compute the dot product of errors and feature
    derivative = np.dot(errors, feature);

    # add L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant:
        derivative = derivative - 2 * l2_penalty * coefficient;

    return derivative;


def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment == +1);
    scores = np.dot(feature_matrix, coefficients);

    lp = np.sum((indicator - 1) * scores - np.log(1. + np.exp(-scores))) - l2_penalty * np.sum(coefficients[1:] ** 2);

    return lp;


def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients, step_size, l2_penalty, max_iter):
    coefficients = np.array(initial_coefficients)  # make sure it's a numpy array
    for itr in xrange(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        ## YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients);

        # Compute indicator value for (y_i = +1)
        indicator = (sentiment == +1);

        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)):  # loop over each coefficient
            is_intercept = (j == 0)
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            derivative = feature_derivative_with_L2(errors, feature_matrix[:, j], coefficients[j], l2_penalty,
                                                    is_intercept);

            # add the step size times the derivative to the current coefficient
            coefficients[j] = coefficients[j] + step_size * derivative;

        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
                or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                  (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients;


# run with L2 = 0
coefficients_0_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                     initial_coefficients=np.zeros(194),
                                                     step_size=5e-6, l2_penalty=0, max_iter=501);

# run with L2 = 4
coefficients_4_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                     initial_coefficients=np.zeros(194),
                                                     step_size=5e-6, l2_penalty=4, max_iter=501);

# run with L2 = 10
coefficients_10_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                      initial_coefficients=np.zeros(194),
                                                      step_size=5e-6, l2_penalty=10, max_iter=501);

# run with L2 = 1e2
coefficients_1e2_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e2, max_iter=501);

# run with L2 = 1e3
coefficients_1e3_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e3, max_iter=501);

# run with L2 = 1e5
coefficients_1e5_penalty = logistic_regression_with_L2(feature_matrix_train, sentiment_train,
                                                       initial_coefficients=np.zeros(194),
                                                       step_size=5e-6, l2_penalty=1e5, max_iter=501);

table = graphlab.SFrame({'word': ['(intercept)'] + important_words})


def add_coefficients_to_table(coefficients, column_name):
    table[column_name] = coefficients;
    return table;


add_coefficients_to_table(coefficients_0_penalty, 'coefficients [L2=0]');
add_coefficients_to_table(coefficients_4_penalty, 'coefficients [L2=4]');
add_coefficients_to_table(coefficients_10_penalty, 'coefficients [L2=10]');
add_coefficients_to_table(coefficients_1e2_penalty, 'coefficients [L2=1e2]');
add_coefficients_to_table(coefficients_1e3_penalty, 'coefficients [L2=1e3]');
add_coefficients_to_table(coefficients_1e5_penalty, 'coefficients [L2=1e5]');

coefficients = list(coefficients_0_penalty[1:]);  # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)];
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x: x[1], reverse=True);

positive_words = word_coefficient_tuples[0:5]
negative_words = word_coefficient_tuples[-5:]
positive_words = ['love','loves', 'easy', 'perfect', 'great'];
negative_words = ['returned', 'waste', 'return', 'money','disappointed'];
print positive_words;
print negative_words
plt.rcParams['figure.figsize'] = 10, 6


def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')

    xx = l2_penalty_list
    plt.plot(xx, [0.] * len(xx), '--', lw=1, color='k')

    table_positive_words = table.filter_by(column_name='word', values=positive_words)
    table_negative_words = table.filter_by(column_name='word', values=negative_words)
    del table_positive_words['word']
    del table_negative_words['word']

    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8 * ((i + 1) / (len(positive_words) * 1.2) + 0.15))
        plt.plot(xx, table_positive_words[i:i + 1].to_numpy().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)

    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8 * ((i + 1) / (len(negative_words) * 1.2) + 0.15))
        plt.plot(xx, table_negative_words[i:i + 1].to_numpy().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)

    plt.legend(loc='best', ncol=3, prop={'size': 16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout();


make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5]);


def get_classification_accuracy(feature_matrix, sentiment, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    apply_threshold = np.vectorize(lambda x: 1. if x > 0  else -1.)
    predictions = apply_threshold(scores)

    num_correct = (predictions == sentiment).sum()
    accuracy = num_correct / len(feature_matrix)
    return accuracy;


train_accuracy = {}
train_accuracy[0] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_0_penalty)
train_accuracy[4] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_4_penalty)
train_accuracy[10] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_10_penalty)
train_accuracy[1e2] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e2_penalty)
train_accuracy[1e3] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e3_penalty)
train_accuracy[1e5] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e5_penalty)

validation_accuracy = {}
validation_accuracy[0] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_0_penalty)
validation_accuracy[4] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_4_penalty)
validation_accuracy[10] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_10_penalty)
validation_accuracy[1e2] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e2_penalty)
validation_accuracy[1e3] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e3_penalty)
validation_accuracy[1e5] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e5_penalty);

# Build a simple report
for key in sorted(validation_accuracy.keys()):
    print "L2 penalty = %g" % key
    print "train accuracy = %s, validation_accuracy = %s" % (train_accuracy[key], validation_accuracy[key])
    print "--------------------------------------------------------------------------------";
