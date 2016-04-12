import graphlab;
import numpy as np;
from math import sqrt;

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


# the [] around 'sqft_living' makes it a list
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price');

print example_features[0, :];  # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0];  # and the corresponding output

my_weights = np.array([1., 1.]);  # the example weights
my_features = example_features[0,];  # we'll use the first data point
predicted_value = np.dot(my_features, my_weights);
print predicted_value;


def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a
    # corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)

    return predictions;


test_predictions = predict_output(example_features, my_weights)

print test_predictions[0];  # should be 1181.0
print test_predictions[1];  # should be 2571.0


def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    return 2 * np.dot(errors, feature);


(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price');
my_weights = np.array([0., 0.]);  # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights);

# just like SFrames 2 numpy arrays can be elementwise subtracted with '-':
errors = test_predictions - example_output;  # prediction errors in this case is just the -example_output

# let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
feature = example_features[:, 0];
derivative = feature_derivative(errors, feature);
print derivative;
print -np.sum(example_output) * 2;  # should be the same as derivative


# recall that the magnitude/length of a vector [g[0], g[1], g[2]] is sqrt(g[0]^2 + g[1]^2 + g[2]^2)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False;
    weights = np.array(initial_weights);  # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights);

        # compute the errors as predictions - output
        prediction_errors = predictions - output;

        gradient_sum_squares = 0;  # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)):  # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            dw = feature_derivative(prediction_errors, feature_matrix[:, i]);

            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += (dw * dw);

            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size * dw;

        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares);
        if gradient_magnitude < tolerance:
            converged = True;
    return weights;


train_data, test_data = sales.random_split(.8, seed=0);

# let's test out the gradient descent
simple_features = ['sqft_living'];
my_output = 'price';
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output);
initial_weights = np.array([-47000., 1.]);
step_size = 7e-12;
tolerance = 2.5e7;

model1 = regression_gradient_descent(simple_feature_matrix, output, initial_weights, step_size, tolerance);
print ("Quiz: Value of weight for sqft_living: %.1f" % model1[1]);

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output);

model1_predictions = predict_output(test_simple_feature_matrix, model1);
print ("Quiz: What is the predicted price for the 1st house in the TEST data set for model 1: %.f" %
       model1_predictions[0]);
print "Test data price: " + str(test_data['price'][0]);
print "Error: " + str(test_output[0] - model1_predictions[0]);

model1_errors = test_output - model1_predictions;
model1_RSS = (model1_errors * model1_errors).sum();
print "Quiz: Residual Sum Of Squares (RSS):" + str(model1_RSS);

# sqft_living15 is the average squarefeet for the nearest 15 neighbors.
model_features = ['sqft_living', 'sqft_living15'];
my_output = 'price';
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output);
initial_weights = np.array([-100000., 1., 1.]);
step_size = 4e-12;
tolerance = 1e9;

model2_weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance);
print "Model 2: intercept:" + str(model2_weights[0]);
print "Model 2: sqft_living weight:" + str(model2_weights[1]);
print "Model 2: sqft_living15 weight:" + str(model2_weights[2]);

(test_feature_matrix, test_feature_output) = get_numpy_data(test_data, model_features, my_output);


model2_predictions = predict_output(test_feature_matrix, model2_weights);


print (
    "Quiz: What is the predicted price for the 1st house in the TEST data set for model 2 (round to nearest dollar)? %.f" %
    model2_predictions[0]);

print "Error: " + str(test_feature_output[0] - model2_predictions[0]);

model2_errors = test_feature_output - model2_predictions;
model2_RSS = (model2_errors * model2_errors).sum();
print "Quiz: Model 2 Residual Sum Of Squares (RSS):" + str(model2_RSS);

print "Question 1: 281.9";
print "Question 2: ";
print "Question 3: ";
print "Question 4: Model 1";
print "Question 5: Model 2";
