import graphlab;
import numpy as np;
import matplotlib.pyplot as plt;
from math import log, sqrt

from docutils.nodes import transition

sales = graphlab.SFrame('../data/kc_house_data.gl/');

sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt);
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt);
sales['bedrooms_square'] = sales['bedrooms'] * sales['bedrooms'];

# In the dataset, 'floors' was defined with type string,
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float);
sales['floors_square'] = sales['floors'] * sales['floors'];

all_features = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated'];

model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None,
                                              l2_penalty=0., l1_penalty=1e10, verbose=False);

model_all_coeff = model_all.get("coefficients");
print model_all_coeff[model_all_coeff['value'] != 0].print_rows();

(training_and_validation, testing) = sales.random_split(.9, seed=1);  # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1);  # split training into train and validate

# l1_penalties = [10^1, 10^1.5, 10^2, 10^2.5,10^3,10^3.5,10^4,10^4.5,10^5,10^5.5,10^6,10^6.5, 10^7];
l1_penalties = np.logspace(1, 7, num=13);

rss_list = [];
min_rss = 0;
best_l1_penalty = None;
best_model = None;
for l1_penalty in l1_penalties:
    m = graphlab.linear_regression.create(training, target='price', features=all_features,
                                          validation_set=None,
                                          l2_penalty=0., l1_penalty=l1_penalty, verbose=False);
    predictions = m.predict(validation);
    rss = sum((validation['price'] - predictions) ** 2);
    if (min_rss > rss or min_rss == 0):
        min_rss = rss;
        best_l1_penalty = l1_penalty;
        best_model = m;

print "Quiz: Best L1_penalty: " + str(best_l1_penalty);
model = graphlab.linear_regression.create(training, target='price', features=all_features,
                                          validation_set=None,
                                          l2_penalty=0., l1_penalty=best_l1_penalty, verbose=False);
test_predictions = best_model.predict(testing);
rss = sum((testing['price'] - test_predictions) ** 2);
model_coeff = best_model .get("coefficients");
print "Quiz: Non-zero weights: " + str(len(model_coeff[model_coeff['value'] != 0]));
model_coeff[model_coeff['value'] != 0].print_rows(num_rows=len(model_coeff));

max_nonzeros = 7;
l1_penalty_values = np.logspace(8, 10, num=20);

l1_penalty_min = 0;
l1_penalty_max = 0;
for l1_penalty in l1_penalty_values:
    m = graphlab.linear_regression.create(training, target='price', features=all_features,
                                          validation_set=None,
                                          l2_penalty=0., l1_penalty=l1_penalty, verbose=False);
    nnz = m['coefficients']['value'].nnz();
    if (l1_penalty_min < l1_penalty and nnz > max_nonzeros):
        l1_penalty_min = l1_penalty;
    if (l1_penalty_max > l1_penalty and nnz < max_nonzeros):
        l1_penalty_max = l1_penalty;
    # Stupid checks on l1_penalty_min and max to handle first iteration.
    if (l1_penalty_min == 0 and nnz > max_nonzeros):
        l1_penalty_min = l1_penalty;
    if (l1_penalty_max == 0 and nnz < max_nonzeros):
        l1_penalty_max = l1_penalty;

print "Quiz: l1_penalty_min: " + str(l1_penalty_min);
print "Quiz: l1_penalty_max: " + str(l1_penalty_max);

l1_penalty_values = np.linspace(l1_penalty_min, l1_penalty_max, 20);

min_rss = 0;
best_model = None;
best_l1_penalty = None;
for l1_penalty in l1_penalty_values:
    m = graphlab.linear_regression.create(training, target='price', features=all_features,
                                          validation_set=None,
                                          l2_penalty=0., l1_penalty=l1_penalty, verbose=False);
    predictions = m.predict(validation);
    rss = sum((validation['price'] - predictions) ** 2);
    nnz = m['coefficients']['value'].nnz();
    # Stupid check on min_rss==0 to handle first iteration.
    if (nnz == max_nonzeros and (min_rss > rss or min_rss == 0)):
        min_rss = rss;
        best_model = m;
        best_l1_penalty = l1_penalty;

print "Quiz: L1_penalty with lowest RSS and sparsity equal to max_nonzeroes: " + str(best_l1_penalty);
model_coeff = best_model.get("coefficients");
print "Quiz: Non-zero weights: " + str(len(model_coeff[model_coeff['value'] != 0]));
model_coeff[model_coeff['value'] != 0].print_rows(num_rows=len(model_coeff));

quit();
