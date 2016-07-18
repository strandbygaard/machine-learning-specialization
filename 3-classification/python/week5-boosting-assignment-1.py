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
target = 'safe_loans';
features = ['grade',  # grade of the loan (categorical)
            'sub_grade_num',  # sub-grade of the loan as a number from 0 to 1
            'short_emp',  # one year or less of employment
            'emp_length_num',  # number of years of employment
            'home_ownership',  # home_ownership status: own, mortgage or rent
            'dti',  # debt to income ratio
            'purpose',  # the purpose of the loan
            'payment_inc_ratio',  # ratio of the monthly payment to income
            'delinq_2yrs',  # number of delinquincies
            'delinq_2yrs_zero',  # no delinquincies in last 2 years
            'inq_last_6mths',  # number of creditor inquiries in last 6 months
            'last_delinq_none',  # has borrower had a delinquincy
            'last_major_derog_none',  # has borrower had 90 day or worse rating
            'open_acc',  # number of open credit accounts
            'pub_rec',  # number of derogatory public records
            'pub_rec_zero',  # no derogatory public records
            'revol_util',  # percent of available credit being used
            'total_rec_late_fee',  # total late fees received to day
            'int_rate',  # interest rate of the loan
            'total_rec_int',  # interest received to date
            'annual_inc',  # annual income of borrower
            'funded_amnt',  # amount committed to the loan
            'funded_amnt_inv',  # amount committed by investors for the loan
            'installment',  # monthly payment owed by the borrower
            ];
loans, loans_with_na = loans[[target] + features].dropna_split();

# Count the number of rows with missing data
num_rows_with_na = loans_with_na.num_rows();
num_rows = loans.num_rows();
print 'Dropping %s observations; keeping %s ' % (num_rows_with_na, num_rows);

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Undersample the safe loans.
percentage = len(risky_loans_raw) / float(len(safe_loans_raw));
safe_loans = safe_loans_raw.sample(percentage, seed=1);
risky_loans = risky_loans_raw;
loans_data = risky_loans.append(safe_loans);

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data));
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data));
print "Total number of loans in our new dataset :", len(loans_data);

train_data, validation_data = loans_data.random_split(.8, seed=1);

model_5 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
                                                   target=target, features=features, max_iterations=5);

# Select all positive and negative examples.
validation_safe_loans = validation_data[validation_data[target] == 1];
validation_risky_loans = validation_data[validation_data[target] == -1];

# Select 2 examples from the validation set for positive & negative loans
sample_validation_data_risky = validation_risky_loans[0:2];
sample_validation_data_safe = validation_safe_loans[0:2];

# Append the 4 examples into a single dataset
sample_validation_data = sample_validation_data_safe.append(sample_validation_data_risky);
sample_validation_data;

model_5_predictions = model_5.predict(sample_validation_data)
correct = sample_validation_data['safe_loans'] == model_5_predictions;
print "Quiz question: What percentage of the predictions on sample_validation_data did model_5 get correct?";
print "%s" % str(sum(correct) / float(len(sample_validation_data)) * 100) + "%";

model_5_probabilities = model_5.predict(sample_validation_data, output_type="probability");
print model_5_probabilities;

print model_5.evaluate(validation_data)['accuracy'];

print "Quiz question: What is the number of false positives on the validation_data?";
predictions = model_5.predict(validation_data)
temp = predictions[validation_data['safe_loans'] == -1];
false_pos = sum(temp == 1)
print false_pos;

print "Number of false negatives:"
temp = predictions[validation_data['safe_loans'] == 1];
false_neg = sum(temp == -1)
print false_neg;

cost = false_neg * 10000 + false_pos * 20000
diff = 49420000 - cost
print diff;

model_5_probability = model_5.predict(validation_data, output_type="probability");
validation_data['predictions'] = model_5_probability;
print validation_data['predictions'].head(5);

print "Quiz question: What grades are the top 5 loans?";
sorted = validation_data.sort('predictions', ascending=False);
print sorted['grade'].head(5);

sorted = validation_data.sort('predictions');
print sorted['grade'].head(5);

model_10 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
                                                    target=target, features=features, max_iterations=10, verbose=False);

model_50 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
                                                    target=target, features=features, max_iterations=50, verbose=False);
model_100 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
                                                     target=target, features=features, max_iterations=100,
                                                     verbose=False);
model_200 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
                                                     target=target, features=features, max_iterations=200,
                                                     verbose=False);
model_500 = graphlab.boosted_trees_classifier.create(train_data, validation_set=None,
                                                     target=target, features=features, max_iterations=500,
                                                     verbose=False);

print model_10.evaluate(validation_data)['accuracy'];
print model_50.evaluate(validation_data)['accuracy'];
print model_100.evaluate(validation_data)['accuracy'];
print model_200.evaluate(validation_data)['accuracy'];
print model_500.evaluate(validation_data)['accuracy'];


def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size': 15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()


train_err_10 = 1 - model_10.evaluate(train_data)['accuracy'];
train_err_50 = 1 - model_50.evaluate(train_data)['accuracy'];
train_err_100 = 1 - model_100.evaluate(train_data)['accuracy'];
train_err_200 = 1 - model_200.evaluate(train_data)['accuracy'];
train_err_500 = 1 - model_500.evaluate(train_data)['accuracy'];

training_errors = [train_err_10, train_err_50, train_err_100,
                   train_err_200, train_err_500]

validation_err_10 = 1 - model_10.evaluate(validation_data)['accuracy'];
validation_err_50 = 1 - model_50.evaluate(validation_data)['accuracy'];
validation_err_100 = 1 - model_100.evaluate(validation_data)['accuracy'];
validation_err_200 = 1 - model_200.evaluate(validation_data)['accuracy'];
validation_err_500 = 1 - model_500.evaluate(validation_data)['accuracy'];

validation_errors = [validation_err_10, validation_err_50, validation_err_100,
                     validation_err_200, validation_err_500];

plt.plot([10, 50, 100, 200, 500], training_errors, linewidth=4.0, label='Training error');
plt.plot([10, 50, 100, 200, 500], validation_errors, linewidth=4.0, label='Validation error');

make_figure(dim=(10, 5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best');
quit();
