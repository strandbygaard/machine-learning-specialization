import graphlab

'''Check GraphLab Create version'''
from distutils.version import StrictVersion
from em_utilities import *
from sklearn.cluster import KMeans

assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'

if __name__ == '__main__':
    wiki = graphlab.SFrame('../../1-foundation/data/people_wiki.gl/').head(5000)
    wiki['tf_idf'] = graphlab.text_analytics.tf_idf(wiki['text'])

    tf_idf, map_index_to_word = sframe_to_scipy(wiki, 'tf_idf')

    tf_idf = normalize(tf_idf)

    for i in range(5):
        doc = tf_idf[i]
        print(np.linalg.norm(doc.todense()))

    np.random.seed(5)
    num_clusters = 25

    # Use scikit-learn's k-means to simplify workflow
    kmeans_model = KMeans(n_clusters=num_clusters, n_init=5, max_iter=400, random_state=1, n_jobs=-1)
    kmeans_model.fit(tf_idf)
    centroids, cluster_assignment = kmeans_model.cluster_centers_, kmeans_model.labels_

    means = [centroid for centroid in centroids]

    num_docs = tf_idf.shape[0]
    weights = []
    for i in xrange(num_clusters):
        # Compute the number of data points assigned to cluster i:
        num_assigned = len(cluster_assignment[cluster_assignment == i])
        w = float(num_assigned) / num_docs
        weights.append(w)

    covs = []
    for i in xrange(num_clusters):
        member_rows = tf_idf[cluster_assignment == i]
        cov = (member_rows.power(2) - 2 * member_rows.dot(diag(means[i]))).sum(axis=0).A1 / member_rows.shape[0] \
              + means[i] ** 2
        cov[cov < 1e-8] = 1e-8
        covs.append(cov)

    out = EM_for_high_dimension(tf_idf, means, covs, weights, cov_smoothing=1e-10)
    out['loglik']


    # Fill in the blanks
    def visualize_EM_clusters(tf_idf, means, covs, map_index_to_word):
        print('')
        print('==========================================================')

        num_clusters = len(means)
        for c in xrange(num_clusters):
            print('Cluster {0:d}: Largest mean parameters in cluster '.format(c))
            print('\n{0: <12}{1: <12}{2: <12}'.format('Word', 'Mean', 'Variance'))

            # The k'th element of sorted_word_ids should be the index of the word
            # that has the k'th-largest value in the cluster mean. Hint: Use np.argsort().
            sorted_word_ids = np.argsort(means[c])[::-1]

            for i in sorted_word_ids[:5]:
                print '{0: <12}{1:<10.2e}{2:10.2e}'.format(map_index_to_word['category'][i],
                                                           means[c][i],
                                                           covs[c][i])
            print '\n=========================================================='


    '''By EM'''
    visualize_EM_clusters(tf_idf, out['means'], out['covs'], map_index_to_word)

    np.random.seed(5)  # See the note below to see why we set seed=5.
    num_clusters = len(means)
    num_docs, num_words = tf_idf.shape

    random_means = []
    random_covs = []
    random_weights = []

    for k in range(num_clusters):
        # Create a numpy array of length num_words with random normally distributed values.
        # Use the standard univariate normal distribution (mean 0, variance 1).
        # YOUR CODE HERE
        mean = np.random.normal(0, 1, num_words)

        # Create a numpy array of length num_words with random values uniformly distributed between 1 and 5.
        # YOUR CODE HERE
        cov = np.random.uniform(1, 5, num_words)

        # Initially give each cluster equal weight.
        # YOUR CODE HERE
        weight = 1

        random_means.append(mean)
        random_covs.append(cov)
        random_weights.append(weight)

    out_random_init = EM_for_high_dimension(tf_idf, random_means, random_covs, random_weights, cov_smoothing=1e-5)

    visualize_EM_clusters(tf_idf, out_random_init['means'], out_random_init['covs'], map_index_to_word)
