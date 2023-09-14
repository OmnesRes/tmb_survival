import numpy as np

def generate_times(n=100, mean_time=6000, risk=0, prob_censored=.5):
    risk_score = np.full(n, risk)
    baseline_hazard = 1 / mean_time
    scale = baseline_hazard * np.exp(risk_score)
    u = np.random.uniform(low=0, high=1, size=len(risk_score))
    t = -np.log(u) / scale
    c = np.random.uniform(low=min(t), high=np.quantile(t, 1.0 - prob_censored), size=n)
    observed_event = t <= c
    observed_time = np.where(observed_event, t, c)
    return observed_time, observed_event
