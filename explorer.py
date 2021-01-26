
import math
import gensim
import pickle as cPickle
import numpy as np
# from tsne import bh_sne
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


class Exploration(dict):

    def __init__(self, query, labels=[], vectors=[]):
        self.query = query.lower()
        self.parsed_query = {}
        self.labels = labels
        self.vectors = vectors
        self.reduction = []
        self.clusters = []
        self.distances = []
        self.stats = {}

    def reduce(self):
        print('Performing tSNE reduction' +
              'on {} vectors'.format(len(self.vectors)))
        # self.reduction = bh_sne(np.array(self.vectors, dtype=np.float64))
        self.reduction = TSNE().fit_transform(np.array(self.vectors, dtype=np.float64))

    def cluster(self, num_clusters=30):
        clustering = KMeans(n_clusters=num_clusters)
        clustering.fit(self.reduction)
        self.clusters = clustering.labels_
        clustermatrix = []
        reduction = self.reduction.tolist()
        for cluster_id in range(num_clusters):
            clustermatrix.append([reduction[i]
                                  for i in range(len(self.vectors))
                                  if self.clusters[i] == cluster_id])
        self.cluster_centroids = clustering.cluster_centers_.tolist()
        self.cluster_centroids_closest_nodes = []
        for cluster_id in range(num_clusters):
            nodes_for_cluster = clustermatrix[cluster_id]
            centroid = self.cluster_centroids[cluster_id]
            closest_node_to_centroid = self._closest_node(
                centroid, nodes_for_cluster)
            coords = nodes_for_cluster[closest_node_to_centroid]
            node_id = reduction.index(coords)
            self.cluster_centroids_closest_nodes.append(node_id)

    def serialize(self):
        result = {
            'query': self.query,
            'parsed_query': self.parsed_query,
            'labels': self.labels,
            'stats': self.stats
        }
        if len(self.reduction) > 0:
            result['reduction'] = self.reduction.tolist()
        if len(self.distances) > 0:
            result['distances'] = self.distances
        if len(self.clusters) > 0:
            result['clusters'] = self.clusters.tolist()
            result['cluster_centroids'] = self.cluster_centroids
            closest_nodes = self.cluster_centroids_closest_nodes
            result['cluster_centroids_closest_nodes'] = closest_nodes
        return result

    def _closest_node(self, node, nodes):
        nodes = np.asarray(nodes)
        dist_2 = np.sum((nodes - node)**2, axis=1)
        return np.argmin(dist_2)


class Model(object):

    def __init__(self, filename):
        try:
            self.model = gensim.models.Word2Vec.load(filename)
        except cPickle.UnpicklingError:
            load = gensim.models.KeyedVectors.load_word2vec_format
            self.model = load(filename, binary=True, limit=1000000)

    def autocomplete(self, query, limit):
        words = []
        i = 0
        for word in self.model.vocab:
            if word.startswith(query):
                words.append({
                    'word': word,
                    'count': self.model.vocab[word].count})
                i += 1

        words = sorted(words, key=lambda x: x['count'], reverse=True)
        return words[0:limit]

    def compare(self, queries, limit):
        all_words = []
        comparison_words = []
        for query in queries:
            positive, negative = self._parse_query(query)
            comparison_words.append(positive[0])
            words, vectors, distances = self._most_similar_vectors(positive, negative, limit)
            all_words += words

        matrix = []
        labels = []
        for word in all_words:
            coordinates = []
            for word2 in comparison_words:
                distance = self.model.n_similarity([word2], [word])
                coordinates.append(distance)
            matrix.append(coordinates)
            labels.append(word)

        return {'labels': labels, 'comparison': matrix}

    def explore(self, query, limit=100):
        print('Model#explore query={}, limit={}'.format(query, limit))
        exploration = Exploration(query)
        if len(query):
            positive, negative = self._parse_query(query.lower())
            exploration.parsed_query['positive'] = positive
            exploration.parsed_query['negative'] = negative
            labels, vectors, distances = self._most_similar_vectors(positive, negative, limit)
            exploration.labels = labels
            exploration.vectors = vectors
            exploration.distances = distances
        else:
            exploration.labels, exploration.vectors, sample_rate = self._all_vectors(limit)
            exploration.stats['sample_rate'] = sample_rate
        exploration.stats['vocab_size'] = len(self.model.vocab)
        exploration.stats['num_vectors'] = len(exploration.vectors)
        return exploration

    def _most_similar_vectors(self, positive, negative, limit):
        print('Model#_most_similar_vectors' +
              'positive={}, negative={}, limit={}'.format(positive, negative, limit))
        results = self.model.most_similar(positive=positive, negative=negative, topn=limit)
        # results = self.model.similar_by_word(positive[0], topn=limit)
        labels = []
        vectors = []
        distances = []
        for key, distance in results:
            distances.append(distance)
            labels.append(key)
            vectors.append(self.model[key])
        return labels, vectors, distances

    def _parse_query(self, query):
        expressions = query.split(' AND ')
        positive = []
        negative = []
        for expression in expressions:
            if expression.startswith('NOT '):
                negative.append(expression[4:])
            else:
                positive.append(expression)
        return positive, negative

    def _all_vectors(self, limit):
        sample = 1
        if limit > -1:
            sample = int(math.ceil(len(self.model.vocab) / limit))
        sample_rate = float(limit) / len(self.model.vocab)
        print('Model#_most_similar_vectors' +
              'sample={}, sample_rate={}, limit={}'.format(sample, sample_rate, limit))
        labels = []
        vectors = []
        i = 0
        for word in self.model.vocab:
            if (i % sample) == 0:
                vectors.append(self.model[word])
                labels.append(word)
            i += 1
        return labels, vectors, sample_rate
