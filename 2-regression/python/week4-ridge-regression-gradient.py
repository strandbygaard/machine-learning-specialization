import graphlab;
import numpy as np;
import matplotlib.pyplot as plt;

sales = graphlab.SFrame('../data/kc_house_data.gl/');


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


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    if feature_is_constant:
        derivative = 2 * np.dot(errors, feature);
    # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
    else:
        derivative = 2 * np.dot(errors, feature) + 2 * l2_penalty * weight;

    return derivative;


(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price');
my_weights = np.array([1., 10.]);
test_predictions = predict_output(example_features, my_weights);
errors = test_predictions - example_output;  # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:, 1], my_weights[1], 1, False);
print np.sum(errors * example_features[:, 1]) * 2 + 20.;
print '';

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:, 0], my_weights[0], 1, True);
print np.sum(errors) * 2.;


def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty,
                                      max_iterations=100):
    weights = np.array(initial_weights)  # make sure it's a numpy array

    # while not reached maximum number of iterations:
    for mi in xrange(max_iterations):
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights);

        # compute the errors as predictions - output
        errors = predictions - output;

        for i in xrange(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            # Remember: when i=0, you are computing the derivative of the constant!
            feature = feature_matrix[:, i];
            derivative = feature_derivative_ridge(errors, feature_matrix[:, i], weights[i], l2_penalty, (i == 0));
            weights[i] = weights[i] - step_size * derivative;
            # subtract the step size times the derivative from the current weight

    return weights


simple_features = ['sqft_living'];
my_output = 'price';

train_data, test_data = sales.random_split(.8, seed=0);

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output);
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output);

initial_weights = np.array([0., 0.]);
step_size = 1e-12;
max_iterations = 1000;

simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size,
                                                             0.0, max_iterations);

simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, output, initial_weights,
                                                                step_size,
                                                                1e11, max_iterations);

predictions_initial = predict_output(simple_test_feature_matrix, initial_weights);
predictions_no_regularization = predict_output(simple_test_feature_matrix, simple_weights_0_penalty);
predictions_high_regularization = predict_output(simple_test_feature_matrix, simple_weights_high_penalty);

errors_initial = test_output - predictions_initial;
errors_no_regularization = test_output - predictions_no_regularization;
errors_high_regularization = test_output - predictions_high_regularization;

rss_initial = (errors_initial ** 2).sum();
rss_no_regularization = (errors_no_regularization ** 2).sum();
rss_high_regularization = (errors_high_regularization ** 2).sum();

print "Quiz 1: coefficient for sqft_living (no regularization): " + str(simple_weights_0_penalty[1])
print "Quiz 2: coefficient for sqft_living (high regularization): " + str(simple_weights_high_penalty[1])

print "Quiz 3:"
print "RSS (initial): " + str(rss_initial);
print "RSS (no regu): " + str(rss_no_regularization);
print "RSS (high regu): " + str(rss_high_regularization);


model_features = ['sqft_living',
                  'sqft_living15'];  # sqft_living15 is the average squarefeet for the nearest 15 neighbors.
my_output = 'price';
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output);
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output);

initial_weights = np.array([0.0, 0.0, 0.0]);
step_size = 1e-12;
max_iterations = 1000;

multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size,
                                                             0.0, max_iterations);

multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size,
                                                             1e11, max_iterations);

predictions_initial = predict_output(test_feature_matrix, initial_weights);
predictions_no_regularization = predict_output(test_feature_matrix, multiple_weights_0_penalty);
predictions_high_regularization = predict_output(test_feature_matrix, multiple_weights_high_penalty);

errors_initial = test_output - predictions_initial;
errors_no_regularization = test_output - predictions_no_regularization;
errors_high_regularization = test_output - predictions_high_regularization;

rss_initial = (errors_initial ** 2).sum();
rss_no_regularization = (errors_no_regularization ** 2).sum();
rss_high_regularization = (errors_high_regularization ** 2).sum();

print "Quiz 5: coefficient for sqft_living (no regularization): " + str(multiple_weights_0_penalty[1])
print "Quiz 6: coefficient for sqft_living (high regularization): " + str(multiple_weights_high_penalty[1])

print "Quiz 7:"
print "RSS (initial): " + str(rss_initial);
print "RSS (no regu): " + str(rss_no_regularization);
print "RSS (high): " + str(rss_high_regularization);

print "Quiz 8:"
print "Predicted price (no regu): " + str(predictions_no_regularization[0]);
diff = abs(predictions_no_regularization[0] - test_output[0]);
print "Diff: " + str(diff);
print "Predicted price (high regu): " + str(predictions_high_regularization[0]);
diff = abs(predictions_high_regularization[0] - test_output[0]);
print "Diff: " + str(diff);


quit();

plt.plot(simple_feature_matrix, output, 'k.',
         simple_feature_matrix, predict_output(simple_feature_matrix, simple_weights_0_penalty), 'b-',
         simple_feature_matrix, predict_output(simple_feature_matrix, simple_weights_high_penalty), 'r-')

plt.show();
