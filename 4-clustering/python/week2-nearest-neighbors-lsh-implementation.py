import numpy as np
import graphlab
from scipy.sparse import csr_matrix
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt

# %matplotlib inline

'''Check GraphLab Create version'''
from distutils.version import StrictVersion

assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.';

'''compute norm of a sparse vector
   Thanks to: Jaiyam Sharma'''


def norm(x):
    sum_sq = x.dot(x.T);
    norm = np.sqrt(sum_sq);
    return (norm);


wiki = graphlab.SFrame('../../1-foundation/data/people_wiki.gl/');

wiki = wiki.add_row_number();

wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text']);


def sframe_to_scipy(column):
    """
    Convert a dict-typed SArray into a SciPy sparse matrix.

    Returns
    -------
        mat : a SciPy sparse matrix where mat[i, j] is the value of word j for document i.
        mapping : a dictionary where mapping[j] is the word whose values are in column j.
    """
    # Create triples of (row_id, feature_id, count).
    x = graphlab.SFrame({'X1': column});

    # 1. Add a row number.
    x = x.add_row_number();
    # 2. Stack will transform x to have a row for each unique (row, key) pair.
    x = x.stack('X1', ['feature', 'value']);

    # Map words into integers using a OneHotEncoder feature transformation.
    f = graphlab.feature_engineering.OneHotEncoder(features=['feature']);

    # We first fit the transformer using the above data.
    f.fit(x);

    # The transform method will add a new column that is the transformed version
    # of the 'word' column.
    x = f.transform(x);

    # Get the feature mapping.
    mapping = f['feature_encoding'];

    # Get the actual word id.
    x['feature_id'] = x['encoded_features'].dict_keys().apply(lambda x: x[0]);

    # Create numpy arrays that contain the data for the sparse matrix.
    i = np.array(x['id']);
    j = np.array(x['feature_id']);
    v = np.array(x['value']);
    width = x['id'].max() + 1;
    height = x['feature_id'].max() + 1;

    # Create a sparse matrix.
    mat = csr_matrix((v, (i, j)), shape=(width, height));

    return mat, mapping;


start = time.time();
corpus, mapping = sframe_to_scipy(wiki['tf_idf']);
end = time.time();
print end - start;

assert corpus.shape == (59071, 547979);
print 'Check passed correctly!';


def generate_random_vectors(num_vector, dim):
    return np.random.randn(dim, num_vector);


np.random.seed(0);
random_vectors = generate_random_vectors(num_vector=16, dim=547979);
random_vectors.shape;

powers_of_two = (1 << np.arange(15, -1, -1));
index_bits = corpus.dot(random_vectors) >= 0;
index_bits.dot(powers_of_two);


def train_lsh(data, num_vector=16, seed=None):
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)

    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    table = {}

    # Partition data points into bins
    bin_index_bits = (data.dot(random_vectors) >= 0)

    # Encode bin index bits into integers
    bin_indices = bin_index_bits.dot(powers_of_two)

    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i.
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table:
            # If no list yet exists for this bin, assign the bin an empty list.
            table[bin_index] = list();
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        doc_ids = table[bin_index];
        doc_ids.append(data_index);

    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}

    return model;


model = train_lsh(corpus, num_vector=16, seed=143);
table = model['table'];
if 0 in table and table[0] == [39583] and \
                143 in table and table[143] == [19693, 28277, 29776, 30399]:
    print 'Passed!';
else:
    print 'Check your code.';

print "Quiz Question";
print np.array(model['bin_index_bits'][35817], dtype=int);
print model['bin_indices'][35817];  # integer format
model['bin_index_bits'][35817] == model['bin_index_bits'][24478];

print np.array(model['bin_index_bits'][24478], dtype=int);
print model['bin_indices'][24478];
model['bin_index_bits'][35817] == model['bin_index_bits'][24478];

doc_ids = list(model['table'][model['bin_indices'][35817]]);
doc_ids.remove(35817);  # display documents other than Obama

docs = wiki.filter_by(values=doc_ids, column_name='id');
print docs;


def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy / (norm(x) * norm(y))
    return 1 - dist[0, 0]


obama_tf_idf = corpus[35817, :]
biden_tf_idf = corpus[24478, :]

print '================= Cosine distance from Barack Obama';
print 'Barack Obama - {0:24s}: {1:f}'.format('Joe Biden',
                                             cosine_distance(obama_tf_idf, biden_tf_idf));
for doc_id in doc_ids:
    doc_tf_idf = corpus[doc_id, :];
    print 'Barack Obama - {0:24s}: {1:f}'.format(wiki[doc_id]['name'],
                                                 cosine_distance(obama_tf_idf, doc_tf_idf));

num_vector = 16;
search_radius = 3;

for diff in combinations(range(num_vector), search_radius):
    print diff;


def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.

    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document

    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)

    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)

    for different_bits in combinations(range(num_vector), search_radius):
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = 1 - alternate_bits[i];

        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            candidate_set = candidate_set.union(
                table[nearby_bin]);  # YOUR CODE HERE: Update candidate_set with the documents in this bin.

    return candidate_set;


obama_bin_index = model['bin_index_bits'][35817];  # bin index of Barack Obama
candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=0);
if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
    print 'Passed test';
else:
    print 'Check your code';
print 'List of documents in the same bin as Obama: 35817, 21426, 53937, 39426, 50261';

candidate_set = search_nearby_bins(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set);
if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
    print 'Passed test';
else:
    print 'Check your code';


def query(vec, model, k, max_search_radius):
    data = model['data'];
    table = model['table'];
    random_vectors = model['random_vectors'];
    num_vector = random_vectors.shape[1];

    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten();

    # Search nearby bins and collect candidates
    candidate_set = set();
    for search_radius in xrange(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set);

    # Sort candidates by their true distances from the query
    nearest_neighbors = graphlab.SFrame({'id': candidate_set});
    candidates = data[np.array(list(candidate_set)), :];
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten();

    return nearest_neighbors.topk('distance', k, reverse=True), len(candidate_set);


print query(corpus[35817, :], model, k=10, max_search_radius=3)[0].join(wiki[['id', 'name']], on='id').sort('distance');

num_candidates_history = [];
query_time_history = [];
max_distance_from_query_history = [];
min_distance_from_query_history = [];
average_distance_from_query_history = [];

for max_search_radius in xrange(17):
    start = time.time();
    result, num_candidates = query(corpus[35817, :], model, k=10,
                                   max_search_radius=max_search_radius);
    end = time.time();
    query_time = end - start;

    print 'Radius:', max_search_radius
    print result.join(wiki[['id', 'name']], on='id').sort('distance');

    average_distance_from_query = result['distance'][1:].mean();
    max_distance_from_query = result['distance'][1:].max();
    min_distance_from_query = result['distance'][1:].min();

    num_candidates_history.append(num_candidates);
    query_time_history.append(query_time);
    average_distance_from_query_history.append(average_distance_from_query);
    max_distance_from_query_history.append(max_distance_from_query);
    min_distance_from_query_history.append(min_distance_from_query);

plt.figure(figsize=(7, 4.5));
plt.plot(num_candidates_history, linewidth=4);
plt.xlabel('Search radius');
plt.ylabel('# of documents searched');
plt.rcParams.update({'font.size': 16});
plt.tight_layout();

plt.figure(figsize=(7, 4.5));
plt.plot(query_time_history, linewidth=4);
plt.xlabel('Search radius');
plt.ylabel('Query time (seconds)');
plt.rcParams.update({'font.size': 16});
plt.tight_layout();

plt.figure(figsize=(7, 4.5));
plt.plot(average_distance_from_query_history, linewidth=4, label='Average of 10 neighbors');
plt.plot(max_distance_from_query_history, linewidth=4, label='Farthest of 10 neighbors');
plt.plot(min_distance_from_query_history, linewidth=4, label='Closest of 10 neighbors');
plt.xlabel('Search radius');
plt.ylabel('Cosine distance of neighbors');
plt.legend(loc='best', prop={'size': 15});
plt.rcParams.update({'font.size': 16});
plt.tight_layout();


