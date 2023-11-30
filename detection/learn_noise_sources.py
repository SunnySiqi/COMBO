from sklearn.mixture import GaussianMixture as GMM
import numpy as np

def noise_sources_from_GMM(cluster_distribution, thereshold=5.0):
	gmm2 = GMM(n_components=2, covariance_type='tied', tol=1e-6, max_iter=100)
	gmm3 = GMM(n_components=3, covariance_type='tied', tol=1e-6, max_iter=100)
	gmm2.fit(cluster_distribution)
	gmm3.fit(cluster_distribution)
	if gmm2.bic(cluster_distribution) - gmm3.bic(cluster_distribution) < thereshold:
		return []
	# noise_col = list({0,1,2} - set({gmm3.means_.argmax(), gmm3.means_.argmin()}))[0]
	# prob = gmm3.predict_proba(cluster_distribution)
	# prob = prob[:, noise_col]
	# noise_sources = np.array([i for i in range(len(cluster_distribution)) if prob[i] > 0.5])
	clean_col = gmm3.means_.argmin()
	prob = gmm3.predict_proba(cluster_distribution)
	prob = prob[:, clean_col]
	noise_sources = np.array([i for i in range(len(cluster_distribution)) if prob[i] < 0.5])
	return noise_sources