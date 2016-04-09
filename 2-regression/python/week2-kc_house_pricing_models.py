import graphlab;
from math import log;

sales = graphlab.SFrame('data/kc_house_data.gl/');

train_data,test_data = sales.random_split(.8,seed=0);

example_features = ['sqft_living', 'bedrooms', 'bathrooms'];
example_model = graphlab.linear_regression.create(train_data, target='price', features=example_features, validation_set=None);

example_weight_summary = example_model.get("coefficients");
print example_weight_summary;

example_predictions = example_model.predict(train_data);
print example_predictions[0]; # should be 271789.505878


def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data);
    # Then compute the residuals/errors
    residuals = predictions-outcome;
    # Then square and add them up
    rss = (residuals * residuals).sum();
    return rss;

rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price']);
print rss_example_train;  # should be 2.7376153833e+14

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2);
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms'];
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x));
train_data['lat_plus_long'] = train_data['lat'] + train_data['long'];
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2);
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms'];
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x));
test_data['lat_plus_long'] = test_data['lat'] + test_data['long'];

sums = test_data['bedrooms_squared'].sum() + test_data['bed_bath_rooms'].sum() + test_data['log_sqft_living'].sum() + \
    test_data['lat_plus_long'].sum();

arith_mean = sums / (test_data.num_rows() * 4);
print "Quiz: Arithmetic mean (bedrooms_squared): " + str(test_data['bedrooms_squared'].sum() / test_data.num_rows());
print "Quiz: Arithmetic mean (bed_bath_rooms): " + str(test_data['bed_bath_rooms'].sum() / test_data.num_rows());
print "Quiz: Arithmetic mean (log_sqft_living): " + str(test_data['log_sqft_living'].sum() / test_data.num_rows());
print "Quiz: Arithmetic mean (lat_plus_long): " + str(test_data['lat_plus_long'].sum() / test_data.num_rows());

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'];
model_2_features = model_1_features + ['bed_bath_rooms'];
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long'];


model_1 = graphlab.linear_regression.create(train_data, target='price', features=model_1_features, validation_set=None);
model_2 = graphlab.linear_regression.create(train_data, target='price', features=model_2_features, validation_set=None);
model_3 = graphlab.linear_regression.create(train_data, target='price', features=model_3_features, validation_set=None);

print "Model 1:\n" + str(model_1.get("coefficients"));
print "Model 2:\n" + str(model_2.get("coefficients"));
print "Model 3:\n" + str(model_3.get("coefficients"));

print "Model 1: RSS (train): " + str(get_residual_sum_of_squares(model_1, train_data, train_data['price']));
print "Model 2: RSS (train): " + str(get_residual_sum_of_squares(model_2, train_data, train_data['price']));
print "Model 3: RSS (train): " + str(get_residual_sum_of_squares(model_3, train_data, train_data['price']));

print "Model 1: RSS (test): " + str(get_residual_sum_of_squares(model_1, test_data, test_data['price']));
print "Model 2: RSS (test): " + str(get_residual_sum_of_squares(model_2, test_data, test_data['price']));
print "Model 3: RSS (test): " + str(get_residual_sum_of_squares(model_3, test_data, test_data['price']));
