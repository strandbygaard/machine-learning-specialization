import graphlab;
import numpy as np;
import matplotlib.pyplot as plt;


# %matplotlib inline
# from math import sqrt;

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


sales = graphlab.SFrame('../data/kc_house_data.gl/');
sales = sales.sort(['sqft_living', 'price']);

l2_small_penalty = 1e-5;
l2_penalty=1e5;

poly15_data = polynomial_sframe(sales['sqft_living'], 15);
poly15_features = poly15_data.column_names();  # get the name of the features
poly15_data['price'] = sales['price'];  # add price to the data since it's the target

model15 = graphlab.linear_regression.create(poly15_data, target='price', features=poly15_features, validation_set=None,
                                            l2_penalty=l2_small_penalty);

model15_coeff = model15.get("coefficients");
print "Quiz 1:"
print model15_coeff[model15_coeff['name'] == "power_1"];

(semi_split1, semi_split2) = sales.random_split(.5, seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

sets = [set_1, set_2, set_3, set_4];
datas = [];
models = [];
modelsr = [];
coeffs = [];
coeffsr = [];
for s in sets:
    s_data = polynomial_sframe(s['sqft_living'], 15);
    datas.append(s_data);
    s_features = s_data.column_names();
    s_data['price'] = s['price'];
    s_model = graphlab.linear_regression.create(s_data, target='price', features=s_features,
                                                validation_set=None, l2_penalty=l2_small_penalty);
    models.append(s_model);
    coeffs.append(s_model.get("coefficients"));
    # Large L2 penalty
    s_model = graphlab.linear_regression.create(s_data, target='price', features=s_features,
                                            validation_set=None, l2_penalty=l2_penalty);
    modelsr.append(s_model);
    coeffsr.append(s_model.get("coefficients"));

plt.plot(datas[0]['power_1'], datas[0]['price'], '.',
         datas[0]['power_1'], models[0].predict(datas[0]), '-',
         datas[1]['power_1'], models[1].predict(datas[1]), '-',
         datas[2]['power_1'], models[2].predict(datas[2]), '-',
         datas[3]['power_1'], models[3].predict(datas[3]), '-');

#plt.show();

plt.plot(datas[0]['power_1'], datas[0]['price'], '.',
         datas[0]['power_1'], modelsr[0].predict(datas[0]), '-',
         datas[1]['power_1'], modelsr[1].predict(datas[1]), '-',
         datas[2]['power_1'], modelsr[2].predict(datas[2]), '-',
         datas[3]['power_1'], modelsr[3].predict(datas[3]), '-');
#plt.show();

print "Quiz 1:"
print model15_coeff[model15_coeff['name'] == "power_1"];

print "Quiz 2 and 3:";
power_1_values = [];
for c in coeffs:
    power_1_values.append(c[c['name'] == "power_1"]['value'][0]);

print power_1_values;

print "Quiz 4 and 5:";
power_1_values = [];
for c in coeffsr:
    power_1_values.append(c[c['name'] == "power_1"]['value'][0]);

print power_1_values;

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)


train_valid_shuffled[0:10] # rows 0 to 9

validation4 = train_valid_shuffled[5818:7757];
print int(round(validation4['price'].mean(), 0));

n = len(train_valid_shuffled);
first_two = train_valid_shuffled[0:2];
last_two = train_valid_shuffled[n-2:n];
print first_two.append(last_two);

part1 = train_valid_shuffled[0:5817];
part2 = train_valid_shuffled[7758:n];
train4 = part1.append(part2);
print int(round(train4['price'].mean(), 0));

#def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):

quit();
