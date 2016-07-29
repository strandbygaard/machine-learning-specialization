import graphlab
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion

assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'

wiki = graphlab.SFrame('../../1-foundation/data/people_wiki.gl');

wiki['word_count'] = graphlab.text_analytics.count_words(wiki['text']);

model = graphlab.nearest_neighbors.create(wiki, label='name', features=['word_count'],
                                          method='brute_force', distance='euclidean');

model.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10);


def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name];
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word', 'count']);
    return word_count_table.sort('count', ascending=False);


obama_words = top_words('Barack Obama');

barrio_words = top_words('Francisco Barrio');

combined_words = obama_words.join(barrio_words, on='word');

combined_words = combined_words.rename({'count': 'Obama', 'count.1': 'Barrio'});

combined_words.sort('Obama', ascending=False);

common_words = combined_words.sort('Obama', ascending=False)[0:5]['word'];


def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys());
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return set(common_words).issubset(unique_words);


wiki['has_top_words'] = wiki['word_count'].apply(has_top_words);

print len(wiki[wiki['has_top_words'] == True]);

print "...Checkpoint..."
print 'Output from your function:', has_top_words(wiki[32]['word_count']);
print 'Correct output: True';
print 'Also check the length of unique_words. It should be 167';

print 'Output from your function:', has_top_words(wiki[33]['word_count']);
print 'Correct output: False';
print 'Also check the length of unique_words. It should be 188';

busch_words = top_words('George W. Bush');
biden_words = top_words('Joe Biden');
obama_busch = graphlab.toolkits.distances.euclidean(set(obama_words['count']), set(busch_words['count']));
print obama_busch;
obama_biden = graphlab.toolkits.distances.euclidean(set(obama_words['count']), set(biden_words['count']));
print obama_biden;
biden_busch = graphlab.toolkits.distances.euclidean(set(biden_words['count']), set(busch_words['count']));
print biden_busch;

obama_busch_words = obama_words.join(busch_words, on='word');
obama_busch_words = obama_busch_words.rename({'count': 'Obama', 'count.1': 'Busch'});
print obama_busch_words.sort('Obama', ascending=False);

wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['word_count']);

model_tf_idf = graphlab.nearest_neighbors.create(wiki, label='name', features=['tf_idf'],
                                                 method='brute_force', distance='euclidean');

model_tf_idf.query(wiki[wiki['name'] == 'Barack Obama'], label='name', k=10);

def top_words_tf_idf(name):
    row = wiki[wiki['name'] == name];
    word_count_table = row[['tf_idf']].stack('tf_idf', new_column_name=['word','weight']);
    return word_count_table.sort('weight', ascending=False);


obama_tf_idf = top_words_tf_idf('Barack Obama');

schiliro_tf_idf = top_words_tf_idf('Phil Schiliro');

obama_schiliro_words = obama_tf_idf.join(schiliro_tf_idf, on='word');
obama_schiliro_words = obama_schiliro_words.rename({'weight':'Obama', 'weight.1':'Schiliro'});
obama_schiliro_words.sort('Obama', ascending=False);

common_words = obama_schiliro_words.sort('Obama', ascending=False)[0:5]['word'];
quit();
