from __future__ import division;
import graphlab;
import math;
import string;

products = graphlab.SFrame('../../1-foundation/data/amazon_baby.gl/');


def remove_punctuation(text):
    import string;
    return text.translate(None, string.punctuation);


review_without_puctuation = products['review'].apply(remove_punctuation);
products['word_count'] = graphlab.text_analytics.count_words(review_without_puctuation);

products = products[products['rating'] != 3];
print len(products);

products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1);
print products;

train_data, test_data = products.random_split(.8, seed=1);
print len(train_data);
print len(test_data);

sentiment_model = graphlab.logistic_classifier.create(train_data,
                                                      target='sentiment',
                                                      features=['word_count'],
                                                      validation_set=None);

weights = sentiment_model.coefficients;
weights.column_names();

num_positive_weights = len(weights[weights['value'] >= 0]);
num_negative_weights = len(weights[weights['value'] < 0]);

print "Question 1:"
print "Number of positive weights: %s " % num_positive_weights
print "Number of negative weights: %s " % num_negative_weights

sample_test_data = test_data[10:13];
print sample_test_data['rating'];
sample_test_data;

scores = sentiment_model.predict(sample_test_data, output_type='margin');
print scores;


def classPrediction(score):
    if (score > 0):
        return (1);
    return (-1);


print scores.apply(classPrediction);
print "Class predictions according to GraphLab Create:";
print sentiment_model.predict(sample_test_data);


def classProbability(score):
    return (1 / (1 + math.exp(-1 * score)));


print scores.apply(classProbability);
print "Class predictions according to GraphLab Create:";
print sentiment_model.predict(sample_test_data, output_type='probability');

test_data['predictions'] = sentiment_model.predict(test_data, output_type='probability');

print test_data.topk('predictions', 20).print_rows(num_rows=20);

print test_data.topk('predictions', 20, reverse=True).print_rows(num_rows=20);


def get_classification_accuracy(model, data, true_labels):
    # First get the predictions
    ## YOUR CODE HERE
    data['predictions'] = model.predict(data);

    # Compute the number of correctly classified examples
    ## YOUR CODE HERE
    num_correct = len(data[data['predictions'] == true_labels]);

    # Then compute accuracy by dividing num_correct by total number of examples
    ## YOUR CODE HERE
    accuracy = num_correct / len(data);

    return accuracy;


accuracy = get_classification_accuracy(sentiment_model, test_data, test_data['sentiment']);
print "Accuracy: " + str(accuracy);

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves',
                     'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed',
                     'work', 'product', 'money', 'would', 'return'];

len(significant_words);

train_data['word_count_subset'] = train_data['word_count'].dict_trim_by_keys(significant_words, exclude=False);
test_data['word_count_subset'] = test_data['word_count'].dict_trim_by_keys(significant_words, exclude=False);

print train_data[0]['review'];
print train_data[0]['word_count'];

print train_data[0]['word_count_subset'];

simple_model = graphlab.logistic_classifier.create(train_data,
                                                   target='sentiment',
                                                   features=['word_count_subset'],
                                                   validation_set=None);
print simple_model;

print get_classification_accuracy(simple_model, test_data, test_data['sentiment']);

print simple_model.coefficients;

print "Simple model:"
print simple_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21);

print "Sentiment model:"
print sentiment_model.coefficients.sort('value', ascending=False).print_rows(num_rows=21);


sentiment_model_train_classification_accuracy = get_classification_accuracy(sentiment_model, train_data, train_data['sentiment']);
simple_model_train_classification_accuracy = get_classification_accuracy(simple_model, train_data, train_data['sentiment']);

print "Sentiment model TRAIN accuracy: " + str(sentiment_model_train_classification_accuracy);
print "Simple model TRAIN accuracy: " + str(simple_model_train_classification_accuracy);

sentiment_model_test_classification_accuracy = get_classification_accuracy(sentiment_model, test_data, test_data['sentiment']);
simple_model_test_classification_accuracy = get_classification_accuracy(simple_model, test_data, test_data['sentiment']);

print "Sentiment model TEST accuracy: " + str(sentiment_model_test_classification_accuracy);
print "Simple model TEST accuracy: " + str(simple_model_test_classification_accuracy);


print "Majority class (train)"
num_positive  = (train_data['sentiment'] == +1).sum()
num_negative = (train_data['sentiment'] == -1).sum()
print "Positive: " + str(num_positive);
print "Negative: " + str(num_negative);

print "Majority class (test)"
num_positive  = (test_data['sentiment'] == +1).sum()
num_negative = (test_data['sentiment'] == -1).sum()
print "Positive: " + str(num_positive);
print "Negative: " + str(num_negative);

test_data['sentiment'] = 1;

print "Sanity: " + str((test_data['sentiment'] == +1).sum()) + " = " + str(len(test_data));
print get_classification_accuracy(sentiment_model, test_data, test_data['sentiment']);