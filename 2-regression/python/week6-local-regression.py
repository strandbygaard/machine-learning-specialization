import graphlab;
import numpy as np;
import math;
import matplotlib.pyplot as plt;
from math import log, sqrt

from docutils.nodes import transition
from matplotlib.mlab import dist

# from sqlalchemy.orm.scoping import query

sales = graphlab.SFrame('../data/kc_house_data_small.gl/');


# sales['floors'] = sales['floors'].astype(int);


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


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0);
    normalized_features = feature_matrix / norms;
    return (normalized_features, norms);


(train_and_validation, test) = sales.random_split(.8, seed=1);  # initial train/test split
(train, validation) = train_and_validation.random_split(.8,
                                                        seed=1);  # split training set into training and validation sets

feature_list = ['bedrooms',
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
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15'];

features_train, output_train = get_numpy_data(train, feature_list, 'price');
features_test, output_test = get_numpy_data(test, feature_list, 'price');
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price');

features_train, norms = normalize_features(features_train);  # normalize training set features (columns)
features_test = features_test / norms;  # normalize test set by training set norms
features_valid = features_valid / norms;  # normalize validation set by training set norms

query_house = features_test[0];

print "Quiz 1:"
dist = np.sqrt(np.sum((query_house - features_train[9]) ** 2));
print "Euclidian Distance: %8.3f" % (dist);

print "Quiz 2:"
distances = [];
for i in range(10):
    distance = np.sqrt(np.sum((query_house - features_train[i]) ** 2))
    distances.append(distance);

print distances.index(min(distances));


def computeDistances(feature_matrix, q):
    differences = feature_matrix - q;
    return np.sqrt(np.sum(differences ** 2, axis=1));


print "Quiz 3:"
print "What is the (0-based) index of the house in the training set that is closest to this query house?"
query_house = features_test[2];
distances = computeDistances(features_train, query_house);
nearest_neighbor = distances.argmin();
print distances.argmin()

print "Quiz 4:"
print "What is the predicted value of the query house based on 1-nearest neighbor regression?"
print "%.0f" % (output_train[nearest_neighbor]);

print "Quiz 5:"
print "What are the indices of the 4 training houses closest to the query house?"


def k_nearest_neighbors(k, feature_matrix, q):
    distances = computeDistances(feature_matrix, q);
    indices = np.argsort(distances);
    return indices[0:k];


query_house = features_test[2];
indices = k_nearest_neighbors(4, features_train, query_house);
print indices;

print "Quiz 6:"
print "Predict the value of the query house by the simple averaging method."


def predict_knn(k, feature_matrix, output_values, q):
    indices = k_nearest_neighbors(k, feature_matrix, q);
    avg = np.sum(output_values[indices]) / k;
    return avg;


query_house = features_test[2];

avg = (output_train[indices[0]] + output_train[indices[1]] + output_train[indices[2]] + output_train[indices[3]]) / 4;
print avg;
avg = predict_knn(4, features_train, output_train, query_house);
print "%.0f" % (avg);

print "Quiz 7:"
print "What is the predicted value of the house in this query set that has the lowest predicted value?"


def predict_kNN_list(k, feature_matrix, output_values, list):
    count = list.shape[0];
    predictions = [];
    for i in range(count):
        house = list[i];
        predicted_value = predict_knn(k, feature_matrix, output_values, house);
        predictions.append(predicted_value);
    return predictions;


predictions = predict_kNN_list(10, features_train, output_train, features_test[0:10])
print min(predictions);