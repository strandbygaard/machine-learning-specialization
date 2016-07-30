import numpy as np
import graphlab
from scipy.sparse import csr_matrix
from itertools import combinations
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt
#%matplotlib inline

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
