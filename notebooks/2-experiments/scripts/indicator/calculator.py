import numpy as np

class RaCalculator:
    def __init__(self, endpoint, initial_Ra=1.0, model="logistic", params=None):
        self.endpoint = endpoint
        self.Ra = float(initial_Ra)
        self.model = model
        self.params = params or {}
        self.P = self.params.get("P0", 0.1)
        self.history = []

    def update_ra(self, eta=None, C2=None):
        if self.model == "sigmoid":
            alpha = self.params.get("alpha", 3.063)
            beta = self.params.get("beta", 0.5)
            self.Ra = 1 / (1 + np.exp(-alpha * ((self.Ra + C2) - beta)))

        elif self.model == "logistic":
            gamma = self.params.get("gamma", 0.2)
            self.Ra += gamma * C2 * self.Ra * (1 - self.Ra)

        elif self.model == "exponential":
            k = self.params.get("k", 0.5)
            anomaly = max(0, eta - 0.5)
            self.Ra *= np.exp(-k * anomaly)

        elif self.model == "recovery":
            gamma = self.params.get("gamma", 0.02)
            delta = self.params.get("delta", 0.2)
            beta = self.params.get("beta", 0.2)

            anomaly = max(0, eta - beta)

            recovery = gamma * (1 - self.Ra)
            damage = delta * anomaly * self.Ra
            self.Ra += recovery - damage

        elif self.model == "kalman":
            self._update_kalman(eta)

        self.Ra = np.clip(self.Ra, 0, 1)

    def _update_kalman(self, eta):
        # parameters
        Q = self.params.get("Q", 0.005)   # process noise
        R = self.params.get("R", 0.05)    # observation noise
        k = self.params.get("k", 3.0)     # eta sensitivity

        # convert eta → health observation
        z = np.exp(-k * eta)

        # prediction step
        Ra_pred = self.Ra
        P_pred = self.P + Q

        # Kalman gain
        K = P_pred / (P_pred + R)

        # correction step
        self.Ra = Ra_pred + K * (z - Ra_pred)

        # update uncertainty
        self.P = (1 - K) * P_pred

    def record(self, window_id, info):
        row = {
            "endpoint": self.endpoint,
            "window_id": window_id,
            "Ra": self.Ra,
            "model": self.model
        }

        row.update(info)
        self.history.append(row)
