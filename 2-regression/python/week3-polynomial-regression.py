import graphlab;
import numpy as np;
import matplotlib.pyplot as plt;

# %matplotlib inline
# from math import sqrt;

tmp = graphlab.SArray([1., 2., 3.]);
tmp_cubed = tmp.apply(lambda x: x ** 3);
print tmp;
print tmp_cubed;

ex_sframe = graphlab.SFrame();
ex_sframe['power_1'] = tmp;
print ex_sframe;


def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame();
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature;

    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree + 1):
            # first we'll give the column a name:
            name = 'power_' + str(power);
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = feature.apply(lambda x: x ** power);
    return poly_sframe;


print polynomial_sframe(tmp, 3);

sales = graphlab.SFrame('../data/kc_house_data.gl/');

sales = sales.sort(['sqft_living', 'price']);

poly1_data = polynomial_sframe(sales['sqft_living'], 1);
poly1_data['price'] = sales['price'];  # add price to the data since it's the target

model1 = graphlab.linear_regression.create(poly1_data, target='price', features=['power_1'], validation_set=None);

# let's take a look at the weights before we plot
model1.get("coefficients");

plt.plot(poly1_data['power_1'], poly1_data['price'], '.',
         poly1_data['power_1'], model1.predict(poly1_data), '-');

# Second degree polynomial
poly2_data = polynomial_sframe(sales['sqft_living'], 2);
my_features = poly2_data.column_names();  # get the name of the features
poly2_data['price'] = sales['price'];  # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target='price', features=my_features, validation_set=None);

model2.get("coefficients");

plt.plot(poly2_data['power_1'], poly2_data['price'], '.',
         poly2_data['power_1'], model2.predict(poly2_data), '-')

# Cubic polynomial
poly3_data = polynomial_sframe(sales['sqft_living'], 3);
my_features = poly3_data.column_names();  # get the name of the features
poly3_data['price'] = sales['price'];  # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target='price', features=my_features, validation_set=None);

print model3.get("coefficients");

plt.plot(poly3_data['power_1'], poly3_data['price'], '.',
         poly3_data['power_1'], model3.predict(poly3_data), '-')

# 15th order polynomial
poly15_data = polynomial_sframe(sales['sqft_living'], 15);
my_features = poly15_data.column_names();  # get the name of the features
poly15_data['price'] = sales['price'];  # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target='price', features=my_features, validation_set=None);

model15.get("coefficients").print_rows(num_rows=16);

plt.plot(poly15_data['power_1'], poly15_data['price'], '.',
         poly15_data['power_1'], model15.predict(poly15_data), '-')

set1, set2 = sales.random_split(.5, seed=0);
set_1, set_2 = set1.random_split(.5, seed=0);
set_3, set_4 = set2.random_split(.5, seed=0);

set_1_data = polynomial_sframe(sales['sqft_living'], 15);
set_1_features = set_1_data.column_names();
set_1_data['price'] = sales['price'];

set_2_data = polynomial_sframe(sales['sqft_living'], 15);
set_2_features = set_2_data.column_names();
set_2_data['price'] = sales['price'];

set_3_data = polynomial_sframe(sales['sqft_living'], 15);
set_3_features = set_3_data.column_names();
set_3_data['price'] = sales['price'];

set_4_data = polynomial_sframe(sales['sqft_living'], 15);
set_4_features = set_4_data.column_names();
set_4_data['price'] = sales['price'];

set_1_model = graphlab.linear_regression.create(set_1_data, target='price', features=set_1_features,
                                                validation_set=None);
set_1_model.get("coefficients").print_rows(num_rows=16);
set_2_model = graphlab.linear_regression.create(set_2_data, target='price', features=set_2_features,
                                                validation_set=None);
set_2_model.get("coefficients").print_rows(num_rows=16);
set_3_model = graphlab.linear_regression.create(set_3_data, target='price', features=set_3_features,
                                                validation_set=None);
set_3_model.get("coefficients").print_rows(num_rows=16);
set_4_model = graphlab.linear_regression.create(set_4_data, target='price', features=set_4_features,
                                                validation_set=None);
set_4_model.get("coefficients").print_rows(num_rows=16);

training_and_validation, testing = sales.random_split(.9, seed=1);
training, validation = training_and_validation.random_split(.5, seed=1);

degree_lowest_rss = 0;
lowest_rss = 1e20;  # initialize it to some value know to be larger than max RSS for the models
for degree in range(1, 15 + 1):
    poly_train = polynomial_sframe(training['sqft_living'], degree);
    poly_validation = polynomial_sframe(validation['sqft_living'], degree);
    features = poly_train.column_names();
    poly_train['price'] = training['price'];
    model = graphlab.linear_regression.create(poly_train, target='price', features=features, validation_set=None,
                                              verbose=False);
    predictions = model.predict(poly_validation);
    errors = (predictions - validation['price']);
    RSS = (errors ** 2).sum();
    if RSS < lowest_rss:
        lowest_rss = RSS;
        degree_lowest_rss = degree;
    print "Degree: " + str(degree) + " RSS: " + str(RSS);

poly_test = polynomial_sframe(testing['sqft_living'], degree_lowest_rss);
features = poly_test.column_names();
poly_test['price'] = testing['price'];
model = graphlab.linear_regression.create(poly_test, target='price', features=features, validation_set=None,
                                          verbose=False);

predictions = model.predict(poly_test);
errors = (predictions - testing['price']);
RSS = (errors ** 2).sum();

print RSS;
