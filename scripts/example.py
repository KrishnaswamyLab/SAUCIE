import SAUCIE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)
    load = SAUCIE.Loader(x, shuffle=False)

    saucie = SAUCIE.SAUCIE(x.shape[1], lambda_c=.2, lambda_d=.4)


    saucie.train(load, 100)
    embedding = saucie.get_embedding(load)
    num_clusters, clusters = saucie.get_clusters(load)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=clusters)
    fig.savefig('embedding_by_cluster.png')
