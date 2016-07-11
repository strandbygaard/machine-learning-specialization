from __future__ import division;
import graphlab;
import math;
from math import sqrt;
import json;
import numpy as np;

import matplotlib.pyplot as plt
# %matplotlib inline


import string;

loans = graphlab.SFrame('../data/lending-club-data.gl/');
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1);
loans = loans.remove_column('bad_loans');
features = ['grade',  # grade of the loan
            'term',  # the term of the loan
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'emp_length',  # number of years of employment
            ];
target = 'safe_loans';
loans = loans[features + [target]];

safe_loans_raw = loans[loans[target] == 1];
risky_loans_raw = loans[loans[target] == -1];

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw));
safe_loans = safe_loans_raw.sample(percentage, seed=1);
risky_loans = risky_loans_raw;
loans_data = risky_loans.append(safe_loans);

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data));
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data));
print "Total number of loans in our new dataset :", len(loans_data);

loans_data = risky_loans.append(safe_loans);
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1});
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature);

    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0);

    loans_data.remove_column(feature);
    loans_data.add_columns(loans_data_unpacked);

features = loans_data.column_names();
features.remove('safe_loans');  # Remove the response variable
features;

train_data, validation_set = loans_data.random_split(.8, seed=1);


def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    if len(data) <= min_node_size:
        return True;
    return False;


print "Quiz question: Given an intermediate node with 6 safe loans and 3 risky loans, if the min_node_size parameter is 10, what should the tree learning algorithm do next?";
print "Stop";


def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return (error_before_split - error_after_split);


def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0;

    # Count the number of 1's (safe loans)
    safe = sum(labels_in_node == 1);

    # Count the number of -1's (risky loans)
    risky = sum(labels_in_node == -1);

    # Return the number of mistakes that the majority classifier makes.
    if (risky <= safe):
        return risky;
    else:
        return safe;


# Test case 1
example_labels = graphlab.SArray([-1, -1, 1, 1, 1]);
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test 1 passed!';
else:
    print 'Test 1 failed... try again!';

# Test case 2
example_labels = graphlab.SArray([-1, -1, 1, 1, 1, 1, 1]);
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test 2 passed!';
else:
    print 'Test 2 failed... try again!';

# Test case 3
example_labels = graphlab.SArray([-1, -1, -1, -1, -1, 1, 1]);
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test 3 passed!';
else:
    print 'Test 3 failed... try again!';


def best_splitting_feature(data, features, target):
    best_feature = None  # Keep track of the best feature
    best_error = 10  # Keep track of the best error so far
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))

    # Loop through each feature to consider splitting on that feature
    for feature in features:

        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0];

        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1];

        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target]);

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target]);

        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes + right_mistakes) / len(data);

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature;
            best_error = error;

    return best_feature;  # Return the best feature we found


if best_splitting_feature(train_data, features, 'safe_loans') == 'term. 36 months':
    print 'Test passed!';
else:
    print 'Test failed... try again!';


def create_leaf(target_values):
    # Create a leaf node
    leaf = {'splitting_feature': None,
            'left': None,
            'right': None,
            'is_leaf': True}


def decision_tree_create(data, features, target, current_depth=0,
                         max_depth=10, min_node_size=1,
                         min_error_reduction=0.0):
    remaining_features = features[:]  # Make a copy of the features.

    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."
        return create_leaf(target_values)

    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."
        return create_leaf(target_values)

    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)

    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size):
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values);

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data));

    # Calculate the error after splitting (number of misclassified examples
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target]);
    right_mistakes = intermediate_node_num_mistakes(right_split[target]);
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))

    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values);

    remaining_features.remove(splitting_feature);
    print "Split on feature %s. (%s, %s)" % ( \
        splitting_feature, len(left_split), len(right_split))

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth,
                                     min_node_size, min_error_reduction);

    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth,
                                      min_node_size, min_error_reduction);

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree}  # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1]);
    num_minus_ones = len(target_values[target_values == -1]);

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1;
    else:
        leaf['prediction'] = -1;

    # Return the leaf node
    return leaf;


def count_nodes(tree):
    if tree['is_leaf']:
        return 1;
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right']);


small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=2, min_node_size=10,
                                           min_error_reduction=0.0);

print small_decision_tree;
# quit();

# if count_nodes(small_decision_tree) == 7:
#     print 'Test passed!';
# else:
#     print 'Test failed... try again!';
#     print 'Number of nodes found                :', count_nodes(small_decision_tree);
#     print 'Number of nodes that should be there : 5';

my_decision_tree_new = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=100,
                                            min_error_reduction=0.0);
my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth=6,
                                            min_node_size=0, min_error_reduction=-1);


def classify(tree, x, annotate=False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction'];
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']];
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value);
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate);
        else:
            return classify(tree['right'], x, annotate);


def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))

    # Once you've made the predictions, calculate the classification error and return it
    mistakes = prediction != data[target];
    return sum(mistakes) / float(len(data));


print evaluate_classification_error(my_decision_tree_new, validation_set);
print evaluate_classification_error(my_decision_tree_old, validation_set);

model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth=2, min_node_size=0,
                               min_error_reduction=-1);
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=0,
                               min_error_reduction=-1);
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth=14, min_node_size=0,
                               min_error_reduction=-1);

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, train_data);
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, train_data);
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, train_data);

print "Training data, classification error (model 1):", evaluate_classification_error(model_1, validation_set);
print "Training data, classification error (model 2):", evaluate_classification_error(model_2, validation_set);
print "Training data, classification error (model 3):", evaluate_classification_error(model_3, validation_set);


def count_leaves(tree):
    if tree['is_leaf']:
        return 1;
    return count_leaves(tree['left']) + count_leaves(tree['right']);


print "Model 1 leaves: " + str(count_leaves(model_1));
print "Model 2 leaves: " + str(count_leaves(model_2));
print "Model 3 leaves: " + str(count_leaves(model_3));

model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=0,
                               min_error_reduction=-1);
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=0,
                               min_error_reduction=0);
model_6 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=0,
                               min_error_reduction=5);

print "Validation data, classification error (model 4):", evaluate_classification_error(model_4, validation_set);
print "Validation data, classification error (model 5):", evaluate_classification_error(model_5, validation_set);
print "Validation data, classification error (model 6):", evaluate_classification_error(model_6, validation_set);


print "Model 4 leaves: " + str(count_leaves(model_4));
print "Model 5 leaves: " + str(count_leaves(model_5));
print "Model 6 leaves: " + str(count_leaves(model_6));

model_7 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=0,
                               min_error_reduction=-1);
model_8 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=2000,
                               min_error_reduction=-1);
model_9 = decision_tree_create(train_data, features, 'safe_loans', max_depth=6, min_node_size=50000,
                               min_error_reduction=-1);

print "Validation data, classification error (model 7):", evaluate_classification_error(model_7, validation_set);
print "Validation data, classification error (model 8):", evaluate_classification_error(model_8, validation_set);
print "Validation data, classification error (model 9):", evaluate_classification_error(model_9, validation_set);


print "Model 7 leaves: " + str(count_leaves(model_7));
print "Model 8 leaves: " + str(count_leaves(model_8));
print "Model 9 leaves: " + str(count_leaves(model_9));

quit();


def decision_tree_create(data, features, target, current_depth=0, max_depth=10):
    remaining_features = features[:]  # Make a copy of the features.

    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(data[target]) == 0:  ## YOUR CODE HERE
        print "Stopping condition 1 reached."
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)

    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == 0:  ## YOUR CODE HERE
        print "Stopping condition 2 reached."
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)

        # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:  ## YOUR CODE HERE
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature(data, features, target);

    # Split on the best feature that we found.
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % ( \
        splitting_feature, len(left_split), len(right_split))

    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target]);
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target]);

    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth);
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth);

    return {'is_leaf': False,
            'prediction': None,
            'splitting_feature': splitting_feature,
            'left': left_tree,
            'right': right_tree};


def count_nodes(tree):
    if tree['is_leaf']:
        return 1;
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right']);


small_data_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=3);
if count_nodes(small_data_decision_tree) == 13:
    print 'Test passed!';
else:
    print 'Test failed... try again!';
    print 'Number of nodes found                :', count_nodes(small_data_decision_tree);
    print 'Number of nodes that should be there : 13';

my_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth=6);


def classify(tree, x, annotate=False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
            print "At leaf, predicting %s" % tree['prediction'];
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']];
        if annotate:
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value);
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate);
        else:
            return classify(tree['right'], x, annotate);


print 'Predicted class: %s ' % classify(my_decision_tree, test_data[0]);

classify(my_decision_tree, test_data[0], annotate=True);


def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))

    # Once you've made the predictions, calculate the classification error and return it
    mistakes = prediction != data[target];
    return sum(mistakes) / float(len(data));


print evaluate_classification_error(my_decision_tree, test_data);


def print_stump(tree, name='root'):
    split_name = tree['splitting_feature']  # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
          % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
             ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'));


print_stump(my_decision_tree);

print "Exploring the intermediate left subtree";
print_stump(my_decision_tree['left'], my_decision_tree['splitting_feature']);

print "Exploring the left subtree of the left subtree";
print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature']);

print "##################"
print "##################"
print "##################"
print "##################"

print_stump(my_decision_tree);

print "Exploring the intermediate right subtree";
print_stump(my_decision_tree['right'], my_decision_tree['splitting_feature']);

print "Exploring the right subtree of the right subtree";
print_stump(my_decision_tree['right']['right'], my_decision_tree['right']['splitting_feature']);

quit();
