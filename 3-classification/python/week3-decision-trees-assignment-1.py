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
# loans.column_names();
# loans['grade'].show();
# loans['home_ownership'].show();

# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x == 0 else -1);
loans = loans.remove_column('bad_loans');

features = ['grade',  # grade of the loan
            'sub_grade',  # sub-grade of the loan
            'short_emp',  # one year or less of employment
            'emp_length_num',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'dti',  # debt to income ratio
            'purpose',  # the purpose of the loan
            'term',  # the term of the loan
            'last_delinq_none',  # has borrower had a delinquincy
            'last_major_derog_none',  # has borrower had 90 day or worse rating
            'revol_util',  # percent of available credit being used
            'total_rec_late_fee',  # total late fees received to day
            ];

target = 'safe_loans';

# Extract the feature columns and target column
loans = loans[features + [target]];

safe_loans_raw = loans[loans[target] == +1];
risky_loans_raw = loans[loans[target] == -1];
print "Number of safe loans  : %s" % len(safe_loans_raw);
print "Number of risky loans : %s" % len(risky_loans_raw);

safe_count = len(safe_loans_raw);
risky_count = len(risky_loans_raw);
print "Percentage of safe loans: %s" % str(safe_count / (safe_count + risky_count));
print "Percentage of risky loans: %s" % str(risky_count / (safe_count + risky_count));

# Since there are fewer risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw));

risky_loans = risky_loans_raw;
safe_loans = safe_loans_raw.sample(percentage, seed=1);

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans);

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data));
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data));
print "Total number of loans in our new dataset :", len(loans_data);

train_data, validation_data = loans_data.random_split(.8, seed=1);

decision_tree_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                                               target=target, features=features);

small_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                                       target=target, features=features, max_depth=2);

validation_safe_loans = validation_data[validation_data[target] == 1];
validation_risky_loans = validation_data[validation_data[target] == -1];

sample_validation_data_risky = validation_risky_loans[0:2];
sample_validation_data_safe = validation_safe_loans[0:2];

sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky);
sample_validation_data;

predicted = decision_tree_model.predict(sample_validation_data);
correct = predicted == sample_validation_data['safe_loans'];
print "Quiz Question: What percentage of the predictions on sample_validation_data did decision_tree_model get correct?";
print sum(correct) / len(correct);

probabilities = decision_tree_model.predict(sample_validation_data, output_type='probability');
print probabilities;

print "Quiz Question: Notice that the probability preditions are the exact same for the 2nd and 3rd loans. Why would this happen?";
probabilities = small_model.predict(sample_validation_data, output_type='probability');
print probabilities;

print "Quiz Question: What is the accuracy of decision_tree_model on the validation set, rounded to the nearest .01?";
print "decision tree model: %s" % decision_tree_model.evaluate(validation_data)['accuracy'];
print "Small model: %s" % small_model.evaluate(validation_data)['accuracy'];

big_model = graphlab.decision_tree_classifier.create(train_data, validation_set=None,
                                                     target=target, features=features, max_depth=10);

print big_model.evaluate(train_data)['accuracy'];
print big_model.evaluate(validation_data)['accuracy'];

print "Quiz Question: How does the performance of big_model on the validation set compare to decision_tree_model on the validation set? Is this a sign of overfitting?";
predictions = decision_tree_model.predict(validation_data);

correct = predictions == validation_data['safe_loans'];
print "Correct: %s" % str(sum(correct) / len(correct));
print "False positive count: %s" % str(sum(validation_data[predictions==1]['safe_loans']==-1));
print "False negative count: %s" % str(sum(validation_data[predictions==-1]['safe_loans']==1));
fp = sum(validation_data[predictions==1]['safe_loans']==-1);
fn = sum(validation_data[predictions==-1]['safe_loans']==1);

print "Cost: %s" % str(fp * 20000 + fn * 10000);

