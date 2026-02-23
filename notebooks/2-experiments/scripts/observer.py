import numpy as np
from scipy.stats import poisson
from scipy.stats import entropy
import math

class TrafficFeatureExtractor:
    def sample_expected_distribution(self, max_count, lambda_ep):
        bins = np.arange(0, max_count+1)
        dY = poisson.pmf(bins, mu=lambda_ep)
        dY = dY / dY.sum()
        return dY

    def get_observed_distribution(self, max_count, current_window):
        obs_counts, _ = np.histogram(
            current_window,
            bins=np.arange(0, max_count+2)
        )

        if obs_counts.sum() == 0:
            return np.ones_like(obs_counts) / len(obs_counts)

        dX = obs_counts / obs_counts.sum()
        return dX

    def kl_divergence(self, max_count, current_window, lambda_ep):
        ''' Calculates KL divergence between p and q'''
        SIG_EPS = 1e-10 # avoids division by zero
        dY = self.sample_expected_distribution(max_count, lambda_ep)
        dX = self.get_observed_distribution(max_count, current_window)

        p = np.asarray(dY, dtype=float) + SIG_EPS
        q = np.asarray(dX, dtype=float) + SIG_EPS

        p = p / p.sum()
        q = q / q.sum()

        D = entropy(p, q)
        D_stable = math.log1p(D)
        return D_stable

    def js_divergence(self, max_count, current_window, lambda_ep, eps=1e-12):
        dY = self.sample_expected_distribution(max_count, lambda_ep)
        dX = self.get_observed_distribution(max_count, current_window)

        p = np.asarray(dX) + eps
        q = np.asarray(dY) + eps
        p /= p.sum()
        q /= q.sum()
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    def calculate_D(self, current_window, lambda_ep):
        max_count = max(
            int(current_window.max()),
            int(lambda_ep*3)+5
        )
        return self.kl_divergence(max_count, current_window, lambda_ep)

    def calculate_Delta(self, Xbar, lambda_ep):
        return (Xbar - lambda_ep) / max(lambda_ep, 1)

    def calculate_ZScore(self, Xbar, lambda_ep, seconds_in_window):
        variance = max(lambda_ep, 1) / seconds_in_window
        return (Xbar - lambda_ep) / math.sqrt(variance)

    def extract_traffic_changes(self, current_window, lambda_ep, seconds_in_window):
        Xbar = current_window.mean()

        D = self.calculate_D(current_window, lambda_ep)
        Delta = self.calculate_Delta(Xbar, lambda_ep)
        Z = self.calculate_ZScore(Xbar, lambda_ep, seconds_in_window)
        return D, Delta, Z

class EndpointAnomalySensor:
    def __init__(self, endpoint, lam):
        self.endpoint = endpoint
        self.lam = lam
        self.tfe = TrafficFeatureExtractor()

    def gaussian_membership(self, u, mu=0.0, sigma=1.0):
        """
        Calculates the feature gaussian membership for
        traffic normality

        Args:
            u: traffic feature
            mu: the normality reference for the feature
            sigma: deviation from normality

        Returns:
            float: value in [0.,1.] to assign no
                    anomaly (0.) or huge anomaly (1.)
        """
        return math.exp(-((u-mu)**2) / (2*(sigma**2)))

    def fuzzification(self, current_window, D, Delta, Z):
        """
        Traffic features are fuzzed using gaussian membership

        Args:
            current_window (np.array): request frequency per second array
            D (float): divergence between observed and expected traffic
            Delta (float): area difference between observed and expected traffic
            Z: z-score of mean deviation between observed and expected traffic
        Returns:
            (float, float, float): membership of each feature
        """
        sigma_u = max(1.0, np.std(current_window))  # adaptive width
        fD = self.gaussian_membership(D, mu=0.0, sigma=sigma_u)
        fDelta = self.gaussian_membership(Delta, mu=0.0, sigma=sigma_u)
        fZ = self.gaussian_membership(Z, mu=0.0, sigma=sigma_u)
        return fD, fDelta, fZ

    def anomaly_score(self,fD, fDelta, fZ):
        """
        Calculates anomaly score for traffic observations

        Args:
            fD (float): traffic divergence normality membership
            fDelta (float): traffic area difference normality membership
            fZ (float): z-score of mean deviation normality membership
        """
        fDprime = 1 - fD
        fDelprime = 1 - fDelta
        fZprime = 1 - fZ
        eta = fDelprime + fDprime + fZprime
        return eta/3

    def calculate_eta(self, current_window, seconds_in_window):
        """
        Calculates traffic anomaly score

        Args:
            current_window (np.array): request frequency per second array
            seconds_in_window (int): window size
        Returns:
            dict: traffic feature and anomaly measurements
        """
        D, Delta, Z = self.tfe.extract_traffic_changes(current_window, self.lam, seconds_in_window)
        fD, fDelta, fZ = self.fuzzification(current_window, D, Delta, Z)
        eta = self.anomaly_score(fD, fDelta, fZ)

        return {
            'eta': eta,
            'D': D,
            'Delta': Delta,
            'Z': Z,
            'fDp': 1-fD,
            'fDeltap': 1-fDelta,
            'fZp': 1-fZ
        }