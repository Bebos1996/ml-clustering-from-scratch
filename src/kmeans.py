from centroid import centroid as cr
from cluster import cluster as cl

def optimiz_func(dataset, clusters):
    m = len(dataset.index)
    acc = 0
    for cluster in clusters:
        for case in cluster.assigned:
            partial = cluster.old_centroid.distance(case)
            acc += partial*partial
    return acc / m

def k_means(dataset, n_clusters, class_name):
    #random initialize centroids
    clusters = [cl(cr(dataset, dataset.columns, class_name)) for _ in range(n_clusters)]
    #repeat until no centroids change
    change = True
    while change:
        for cluster in clusters: 
            cluster.assigned = []
        change = False
        for _, row in dataset.iterrows():
            index = 0
            distance = clusters[0].centroid.distance(row)
            for j in range(1, len(clusters)):
                dist = clusters[j].centroid.distance(row)
                if dist < distance :
                    distance = dist
                    index = j
            clusters[index].assigned.append(row)
        clusters = [cluster for cluster in clusters if len(cluster.assigned) > 0]
        for cluster in clusters:
            cluster.new_centroid()
            if cluster.changed == True:
                change = True
        cost = optimiz_func(dataset, clusters)
    return clusters, cost