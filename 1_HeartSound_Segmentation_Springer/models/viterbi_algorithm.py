import numpy as np
from scipy.stats import multivariate_normal
from utils.tools import get_duration

_LOG_TINY = np.log(np.finfo(float).tiny)
# Fixed predecessor map (0-indexed): cyclic state sequence 0→1→2→3→0
_PREV_STATE = [3, 0, 1, 2]
_BOUNDS_KEYS = [
    ('min_s1',       'max_s1'),
    ('min_systole',  'max_systole'),
    ('min_s2',       'max_s2'),
    ('min_diastole', 'max_diastole'),
]


def viterbi_decode_pcg(observation_sequence, pi_vector, model,
                       total_obs_distribution, heart_rate, systolic_time, feature_fs):
    t = len(observation_sequence)
    n = 4
    max_duration_d = round(60 / heart_rate * feature_fs)

    # ------------------------------------------------------------------
    # Observation probabilities  (t × n)  – one predict_proba call each
    # ------------------------------------------------------------------
    observation_probs = np.zeros((t, n))
    po_correction = multivariate_normal.pdf(observation_sequence,
                                            total_obs_distribution[0],
                                            total_obs_distribution[1])
    for i in range(n):
        pihat = model[i].predict_proba(observation_sequence)[:, 1]
        observation_probs[:, i] = (pihat * po_correction) / pi_vector[i]

    # Precompute cumulative log-probs for O(1) emission slices
    # cum_log[k, j] = sum of safe_log[0..k-1, j]
    safe_log_obs = np.log(np.maximum(observation_probs, np.finfo(float).tiny))  # (t, n)
    cum_log_obs = np.zeros((t + 1, n))
    cum_log_obs[1:] = np.cumsum(safe_log_obs, axis=0)

    # ------------------------------------------------------------------
    # Duration probabilities (n × max_duration_d) – vectorised over d
    # ------------------------------------------------------------------
    durations = get_duration(heart_rate, systolic_time, feature_fs)
    d_arr = np.arange(max_duration_d, dtype=float)

    duration_probs = np.zeros((n, max_duration_d))
    duration_sum   = np.zeros(n)

    for j in range(n):
        mean_j = durations['d_distributions'][j, 0]
        var_j  = durations['d_distributions'][j, 1]
        probs  = multivariate_normal.pdf(d_arr, mean_j, var_j)
        lo_key, hi_key = _BOUNDS_KEYS[j]
        lo, hi = int(durations[lo_key]), int(durations[hi_key])
        mask = (d_arr < lo) | (d_arr > hi)
        probs[mask] = np.finfo(float).tiny
        duration_probs[j] = probs
        duration_sum[j]   = probs.sum()

    # log(duration_prob / duration_sum) precomputed
    log_dur_norm = np.log(duration_probs) - np.log(duration_sum[:, np.newaxis])  # (n, D)

    # ------------------------------------------------------------------
    # Viterbi DP
    # ------------------------------------------------------------------
    total_t = t + max_duration_d - 1
    delta        = np.full((total_t, n), -np.inf)
    psi          = np.zeros((total_t, n))
    psi_duration = np.zeros((total_t, n))

    delta[0] = np.log(pi_vector) + safe_log_obs[0]
    psi[0]   = -1

    d_indices = np.arange(max_duration_d)  # (D,)

    for window_t in range(1, total_t):
        end_t = min(window_t, t - 1)  # scalar, fixed for this window_t

        for j in range(n):
            prev_s = _PREV_STATE[j]

            # start_t for every d  – vectorised
            start_t_arr = np.minimum(np.maximum(0, window_t - d_indices), t - 1)  # (D,)

            # Emission log-prob: product of obs probs from start_t to end_t-1
            emission_log = cum_log_obs[end_t, j] - cum_log_obs[start_t_arr, j]  # (D,)
            bad = (emission_log == 0) | ~np.isfinite(emission_log)
            emission_log[bad] = _LOG_TINY

            # Transition: unique predecessor → delta of predecessor at start_t
            delta_from = delta[start_t_arr, prev_s]  # (D,)

            candidates = delta_from + emission_log + log_dur_norm[j]  # (D,)

            best_d = int(np.argmax(candidates))
            best_v = candidates[best_d]

            if best_v > delta[window_t, j]:
                delta[window_t, j]        = best_v
                psi[window_t, j]          = prev_s + 1
                psi_duration[window_t, j] = best_d

    # ------------------------------------------------------------------
    # Back-tracking
    # ------------------------------------------------------------------
    qt = np.zeros((1, total_t))

    temp_delta = delta[t:, :]
    pos, state = divmod(int(np.argmax(temp_delta)), n)
    pos += t

    offset = pos
    preceding_state = psi[offset, state]
    onset = offset - psi_duration[offset, state]
    qt[0, int(onset) - 1:int(offset)] = state + 1

    state = preceding_state
    count = 0
    while onset != 0 and count < 10000:
        offset         = onset - 1
        preceding_state = psi[int(offset) - 1, int(state) - 1]
        onset           = offset - psi_duration[int(offset) - 1, int(state) - 1]
        if onset < 1:
            onset = 1
        qt[0, int(onset) - 1:int(offset)] = state
        state = preceding_state
        count += 1

    qt = qt[0, :t]
    return delta, psi, qt
