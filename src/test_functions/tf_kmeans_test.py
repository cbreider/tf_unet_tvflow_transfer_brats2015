import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_kmeans(points, centroids, iteration_n):
    points_expanded = tf.expand_dims(points, 0)
    for step in range(iteration_n):
        centroids_expanded = tf.expand_dims(centroids, 1)

        distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
        assignments = tf.argmin(distances, 0)

        means = []
        for c in range(clusters_n):
            eq = tf.equal(assignments, c)
            eqw = tf.where(eq)
            eqwr = tf.reshape(eqw, [1, -1])
            eqwrsl = tf.gather(points, eqwr)
            mean = tf.reduce_mean(eqwrsl, reduction_indices=[1])
            means.append(mean)
            #means.append(tf.reduce_mean)
                #tf.gather(points,
                 #         tf.reshape(
                  #            tf.where(
                    #              tf.equal(assignments, c)
                   #           ), [1, -1])
                     #     ), reduction_indices=[1]))

        new_centroids = tf.concat(means, 0)
        centroids = new_centroids
        #update_centroids = tf.assign(centroids, new_centroids)

    centroids_expanded = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)
    return [centroids, assignments]


points_n = 200
clusters_n = 3
iteration_n = 100

points = np.random.uniform(0, 10, (points_n, 2))
centroids = np.random.uniform(0, 10, (clusters_n, 2))


points_pl = tf.placeholder(tf.float32, shape=[points_n, 2])
centroids_pl = tf.placeholder(tf.float32, shape=[clusters_n, 2])
kmeans = get_kmeans(points_pl, centroids_pl, iteration_n)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(iteration_n):
        [centroid_values, assignment_values] = sess.run(kmeans, feed_dict={points_pl: points, centroids_pl: centroids})

    print("centroids", centroid_values)

plt.scatter(points[:, 0], points[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()