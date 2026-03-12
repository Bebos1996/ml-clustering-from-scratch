import numpy as np
import pandas as pd
import sys
import itertools
from kmeans import k_means
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#Metodi per calcolo silhouette
def silhouette_coeff(clusters):
    for cluster in clusters:
        size = len(cluster.assigned)
        if size == 0:
            cluster.silhouettes.append(0)
        else:
            for i in range(0, len(cluster.assigned)):
                a = silh_a_coeff(cluster.assigned[i], cluster.assigned)
                other_clusters = clusters.copy()
                other_clusters.remove(cluster)
                b = silh_b_coeff(cluster.assigned[i], other_clusters)
                if a < b:
                    cluster.silhouettes.append(1-(a/b))
                elif a == b:
                    cluster.silhouettes.append(0)
                else:
                    cluster.silhouettes.append((b/a)-1)

def silh_a_coeff(case, assigned):
    partial = 0
    for inst in assigned:
        dist = 0
        for i in range(0, len(case)-1):
            dist += np.sqrt(np.power(case[i]-inst[i],2))
        partial += dist
    return partial / (len(assigned) - 1)

def silh_b_coeff(case, other_clusters):
    min = 0
    cluster_0 = other_clusters[0]
    for inst in cluster_0.assigned:
        dist = 0
        for i in range(0, len(case)-1):
            dist += np.sqrt(np.power(case[i]-inst[i],2))
        min += dist
    min = min / len(cluster_0.assigned)
    for i in range(1, len(other_clusters)):
        tmp = 0
        for inst in other_clusters[i].assigned:
            dist = 0
            for j in range(0, len(case)-1):
                dist += np.sqrt(np.power(case[j]-inst[j],2))
            tmp += dist
        tmp = tmp / len(other_clusters[i].assigned)
        if tmp < min:
            min = tmp
    return min

def mean_silhouette(clusters):
    partial = 0
    for cluster in clusters:
        for sil in cluster.silhouettes:
            partial += sil
    return partial / len(dataset.index)

def silhouette_plot(dataset_df, clusters, mean_silhouette, input_colors):
    dataset = dataset_df.values
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, ax1 = plt.subplots(1,1)
    fig.set_size_inches(12, 6)
    labels = [cl.label for cl in clusters]
    ax1.set_xlim([-0.5, 1])
    ax1.set_ylim([0, len(dataset) + (len(clusters) + 1) * 10])

    y_lower = 10
    #colors = [cm.nipy_spectral(float(i) / len(clusters)) for i in range(len(clusters))]
    colors2 = cm.get_cmap('nipy_spectral_r', len(clusters))
    colors_vec = [colors2(i) for i in input_colors]
    
    for i in range(len(clusters)):
        ith_cluster_silhouette_values = clusters[i].silhouettes
        ith_cluster_silhouette_values.sort()
        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=colors2(i),
            edgecolor=colors2(i),
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.35, y_lower + 0.5 * size_cluster_i, labels[i])

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot with k = %d" %len(clusters))
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=mean_silhouette, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

#validazione esterna
def assign_clusters_labels(dataset, clusters, class_col_name):
    labels_list = dataset[class_col_name].unique()
    for cluster in clusters:
        max = 0
        label_name = "None"
        for label in labels_list:
            current_len = len(list(filter(lambda row: row[class_col_name] == label, cluster.assigned)))
            if current_len > max:
                max = current_len
                label_name = label
        cluster.label = label_name

def find_index(list_of_lists, class_name):
    for idx, el in enumerate(list_of_lists):
        if el[0] == class_name:
            return idx
    return -1

def incorrect_clustered_cases(clusters, class_col_name):
    total = 0
    assignements = []
    for idx, cluster in enumerate(clusters):
        errors = len(list(filter(lambda row: row[class_col_name] != cluster.label, cluster.assigned)))
        for key, group in itertools.groupby(cluster.assigned, lambda row: row[class_col_name]):
            index = find_index(assignements, key)
            if index == -1:
                assignements.append([key])
                assignements[-1].append((len(list(group)),idx))
            else:
                assignements[index].append((len(list(group)), idx))
        total += errors

    tmp_str = ""
    for i in range(len(clusters)):
        tmp_str += str(i) + '\t'
    tmp_str += '\n'
    
    for sublist in assignements:
        prev = 0
        for idx, el in enumerate(sublist):
            if idx != 0:
                for i in range(prev, len(clusters)):
                    if el[1] == i:
                        tmp_str += str(el[0]) + '\t'
                        prev = i + 1
                        if idx + 1 < len(sublist):
                            break
                    else:
                        tmp_str += "0\t"
        tmp_str += str(sublist[0])
        tmp_str += '\n'
    
    for idx,cluster in enumerate(clusters):
        tmp_str += "cluster " + str(idx) + ": " + str(cluster.label) + '\n'

    print(tmp_str)
    return total

#validazione interna
def purity(clusters, class_col_name):
    total = 0
    for cluster in clusters:
        total += len(list(filter(lambda row: row[class_col_name] == cluster.label, cluster.assigned)))
    return total / len(dataset.index)

def ri_prec_rec_fmeas(clusters, class_col_name):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(0, len(clusters)):
        inner_pairs_list = [(a, b) for idx, a in enumerate(clusters[i].assigned) for b in clusters[i].assigned[idx + 1:]]
        TP += len(list(filter(lambda x: x[0][class_col_name] == x[1][class_col_name], inner_pairs_list)))
        FP += len(list(filter(lambda x: x[0][class_col_name] != x[1][class_col_name], inner_pairs_list)))
        for j in range (i+1, len(clusters)):
            ext_pairs_list = list(itertools.product(clusters[i].assigned, clusters[j].assigned))
            TN += len(list(filter(lambda x: x[0][class_col_name] != x[1][class_col_name], ext_pairs_list)))
            FN += len(list(filter(lambda x: x[0][class_col_name] == x[1][class_col_name], ext_pairs_list)))
    rand_index = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1measure = (2*(precision*recall)) / (precision+recall)
    return (rand_index, precision, recall, f1measure)

def clustering_random_initialization(dataset, n_centers, class_name, iterations):
    best_clusters = []
    best_cost = float("inf")
    for _ in range(iterations):
        clusters, cost = k_means(dataset, n_centers, class_name)
        if cost < best_cost:
            best_cost = cost
            best_clusters = clusters
    return best_clusters, best_cost

def elbow_method(dataset, class_name, iterations, max_centers):
    clusters_list = []
    costs_list = []
    for num_centers in range(1, max_centers + 1):
        print("Calculating elbow method with k = %d" %num_centers)
        clusters, cost = clustering_random_initialization(dataset, num_centers, class_name, iterations)
        costs_list.append(cost)
        clusters_list.append(clusters)
    plt.plot(np.arange(1, max_centers+1), np.array(costs_list))
    plt.xlabel("Numero K")
    plt.ylabel("Funzione di costo")
    plt.title("Elbow method")
    plt.show()
    #substract one to get the correct index in list
    index = int(input("Insert the best K: "))-1
    return index, clusters_list[index], costs_list[index]

def create_colors_vec(clusters):
    l1 = []
    for idx, cluster in enumerate(clusters):
        for el in cluster.assigned:
            l1.append((el.name,idx))
    l1.sort(key=lambda tup: tup[0])
    c = np.array([t[1] for t in l1])
    return c

def create_markers_vec(labels):
    vals = list(labels.unique())
    markers = ['o', '*', 'P']
    markers_vec = []
    for el in labels:
        index = vals.index(el)
        markers_vec.append(markers[index])
    return markers, np.array(markers_vec)

def silhouette_to_best_k(dataset, max_centers, iterations, class_col_name):
    for i in range(2, max_centers + 1):
        clusters, cost = clustering_random_initialization(dataset, i, class_col_name, iterations)
        colors = create_colors_vec(clusters)
        silhouette_coeff(clusters)
        mean_silh = mean_silhouette(clusters)
        assign_clusters_labels(dataset, clusters, class_col_name)
        silhouette_plot(dataset, clusters, mean_silh, colors)

def create_legends_element(markers_unique, class_unique):
    legend_elements = []
    for i in range(len(markers_unique)):
        legend_elements.append(
            Line2D([0],[0],marker=markers_unique[i], label=class_unique[i], markersize=10, color='w', markerfacecolor='black')
        )
    return legend_elements

def visualize_assignments(dataset, input_colors, markers, markers_vec, features, class_col_name):
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(12, 6)
    pairs = [(a, b) for idx, a in enumerate(features) for b in features[idx + 1:]]
    colors = ['r', 'b', 'g', 'y', 'm']
    colors_vec = [colors[i] for i in input_colors]
    index = 0
    for row in range(2):
        for col in range(3):
            xdata_label, ydata_label = pairs[index]
            index = index + 1
            axs[row, col].set_xlabel(xdata_label)
            axs[row, col].set_ylabel(ydata_label)
            for idx, instance in dataset.iterrows():
                axs[row, col].scatter(instance[xdata_label], instance[ydata_label], c=colors_vec[idx], marker=markers_vec[idx])
    fig.legend(handles=create_legends_element(markers,dataset[class_col_name].unique()),loc='upper left')
    fig.suptitle('Assignments', fontsize=16)
    plt.show()

dataset = pd.read_csv("./../data/Iris.csv")
class_col_name = "Class"
cols_to_use = list(filter(lambda x: x != class_col_name, dataset.columns))
#silhouette_to_best_k(dataset, 6, 5, class_col_name)
k, clusters, cost = elbow_method(dataset, class_col_name, 10, 5)
colors = create_colors_vec(clusters)
markers, markers_vec = create_markers_vec(dataset[class_col_name])
visualize_assignments(dataset, colors, markers, markers_vec, cols_to_use, class_col_name)
silhouette_coeff(clusters)
mean_silh = mean_silhouette(clusters)
assign_clusters_labels(dataset, clusters, class_col_name)
print("Incorrect clustered instances: %d" % incorrect_clustered_cases(clusters, class_col_name))
print("Purity: %f" % purity(clusters, class_col_name))
rand, prec, rec, fmeas = ri_prec_rec_fmeas(clusters, class_col_name)
print("Rand index: %f" % rand)
print("Precision: %f" % prec)
print("Recall: %f" % rec)
print("F1measure: %f" % fmeas)
print("Mean silhouette: %f" % mean_silh)
silhouette_plot(dataset, clusters, mean_silh, colors)