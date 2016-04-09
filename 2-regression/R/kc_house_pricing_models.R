train_data = read.csv("data/kc_house_train_data.csv")
test_data = read.csv("data/kc_house_test_data.csv")
train_data$date = as.Date(strptime(train_data$date, "%Y%m%dT%H%M%S"))
test_data$date = as.Date(strptime(test_data$date, "%Y%m%dT%H%M%S"))

hist(train_data$date, breaks=100)
summary(train_data$date)

features = c('sqft_living', 'bedrooms', 'bathrooms');

example_model = lm(price ~ sqft_living + bedrooms + bathrooms, data=train_data)
example_predictions = predict(example_model,newdata=train_data)
example_predictions[1]



train_data$bedrooms_squared = train_data$bedrooms^2;
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms'];
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x));
train_data['lat_plus_long'] = train_data['lat'] + train_data['long'];
test_data$bedrooms_squared = test_data$bedrooms^2;
test_data$bed_bath_rooms = test_data$bedrooms * test_data$bathrooms;
test_data$log_sqft_living = log(test_data$sqft_living);
test_data$lat_plus_long = test_data$lat + test_data$long;

sums = test_data$bedrooms_squared + test_data$bed_bath_rooms + test_data$log_sqft_living + test_data$lat_plus_long;
test_data
sum(sums) / (4* nrow(test_data));






