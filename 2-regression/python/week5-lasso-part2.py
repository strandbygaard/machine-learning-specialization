import graphlab;
import numpy as np;
import math;
import matplotlib.pyplot as plt;
from math import log, sqrt

from docutils.nodes import transition

sales = graphlab.SFrame('../data/kc_house_data.gl/');
# In the dataset, 'floors' was defined with type string,
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int);


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1;  # this is how you add a constant column to an SFrame

    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features;  # this is how you combine two lists

    # select the columns of data_SFrame given by the features list into the SFrame
    # features_sframe (now including constant):
    features_sframe = graphlab.SFrame();
    for feature in features:
        features_sframe[feature] = data_sframe[feature];

    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()

    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = graphlab.SArray(data_sframe[output]);

    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy();

    return feature_matrix, output_array;


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a
    # corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)

    return predictions;


# X = np.array([[3., 5., 8.], [4., 12., 15.]]);
# print X;
#
# norms = np.linalg.norm(X, axis=0);  # gives [norm(X[:,0]), norm(X[:,1]), norm(X[:,2])]
# print norms;
# print X / norms;  # gives [X[:,0]/norm(X[:,0]), X[:,1]/norm(X[:,1]), X[:,2]/norm(X[:,2])]


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0);
    normalized_features = feature_matrix / norms;
    return (normalized_features, norms);


simple_features = ['sqft_living', 'bedrooms'];
my_output = 'price';
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output);
simple_feature_matrix, norms = normalize_features(simple_feature_matrix);

weights = np.array([1., 4., 1.]);
prediction = predict_output(simple_feature_matrix, weights);


def compute_ro(feature_matrix, weights):
    ro = [i for i in range(len(weights))];
    for i in range(len(weights)):
        feature_i = feature_matrix[:, i];
        ro[i] = sum(feature_i * (output - prediction + weights[i] * feature_i));
    return ro;


# ro[i] = SUM[ [feature_i]*(output - prediction + w[i]*[feature_i]) ]
ro = compute_ro(simple_feature_matrix, weights);

print "Quiz 1 and 2:"
print str(ro[1] * 2 / 1e8);
print str(ro[2] * 2 / 1e8);


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = predict_output(feature_matrix, weights);
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    feature_i = feature_matrix[:, i];
    ro_i = sum(feature_i * (output - prediction + weights[i] * feature_i));

    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2;
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2;
    else:
        new_weight_i = 0.;

    return new_weight_i;


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    weights = initial_weights;
    while (True):

        maxchange = 0;
        for i in range(len(weights)):

            old_weights_i = weights[i]  # remember old value of weight[i], as it will be overwritten
            # the following line uses new values for weight[0], weight[1], ..., weight[i-1]
            #     and old values for weight[i], ..., weight[d-1]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty);

            # use old_weights_i to compute change in coordinate
            change = abs(old_weights_i - weights[i]);
            if (change > maxchange):
                maxchange = change;
        if (maxchange < tolerance):
            return weights;


simple_features = ['sqft_living', 'bedrooms'];
my_output = 'price';
initial_weights = np.zeros(3);
l1_penalty = 1e7;
tolerance = 1.0;
(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output);
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix);  # normalize features

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output, initial_weights, l1_penalty,
                                            tolerance);

predictions = predict_output(normalized_simple_feature_matrix, weights);
RSS = sum((predictions - output) ** 2);
print "Quiz 3: RSS: " + str(RSS);
print weights;

train_data, test_data = sales.random_split(.8, seed=0);
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated'];
l1_penalty = 1e7;
tolerance = 1;
(all_feature_matrix, output) = get_numpy_data(train_data, all_features, my_output);
initial_weights = np.zeros(14);
(normalized_all_feature_matrix, all_norms) = normalize_features(all_feature_matrix);  # normalize features
weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty,
                                               tolerance);

print "Quiz 5: ";
print weights1e7[0];
print weights1e7[3];
print weights1e7[9];
print weights1e7[6];
print weights1e7[11];

l1_penalty = 1e8;
tolerance = 1.0;
initial_weights = np.zeros(14);
weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty,
                                               tolerance);

print "Quiz 6: ";
print weights1e8[0];
print weights1e8[3];
print weights1e8[9];
print weights1e8[6];
print weights1e8[11];

l1_penalty = 1e8
tolerance = 1.0;
initial_weights = np.zeros(14);
weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty,
                                               tolerance);

print "Quiz 7: ";

l1_penalty = 1e4;
tolerance = 5e5;
initial_weights = np.zeros(14);
weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output, initial_weights, l1_penalty,
                                               tolerance);
print weights1e4[0];
print weights1e4[3];
print weights1e4[9];
print weights1e4[6];
print weights1e4[11];

weights_normalized_1e4 = weights1e4 / all_norms;
weights_normalized_1e7 = weights1e7 / all_norms;
weights_normalized_1e8 = weights1e8 / all_norms;

print weights_normalized_1e7[3];

(test_feature_matrix, test_output) = get_numpy_data(test_data, all_features, 'price')

predictions1e4 = predict_output(test_feature_matrix, weights_normalized_1e4);
predictions1e7 = predict_output(test_feature_matrix, weights_normalized_1e7);
predictions1e8 = predict_output(test_feature_matrix, weights_normalized_1e8);
rss1e4 = sum((predictions1e4 - test_output) ** 2);
rss1e7 = sum((predictions1e7 - test_output) ** 2);
rss1e8 = sum((predictions1e8 - test_output) ** 2);

print "Quiz 8:";
print rss1e4;
print rss1e7;
print rss1e8;
