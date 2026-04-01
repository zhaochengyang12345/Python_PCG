import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import correlate as _scipy_correlate

from analytics.homomorphic_envelope import homomorphic_envelope
from analytics.hilbert_envelope import hilbert_envelope
from analytics.wavelet_envelope import wavelet_envelope
from utils.preprocessing import schmidt_spike_removal, normalize, \
    high_pass_filter, downsample, low_pass_filter, wavelet_denoise

# State display configuration
STATE_COLORS = {1: '#e74c3c', 2: '#f39c12', 3: '#2ecc71', 4: '#3498db'}
STATE_LABELS = {1: 'S1', 2: 'Systole', 3: 'S2', 4: 'Diastole'}


def label_pcg_states(envelope, s1, s2, signal, feature_fs):
    states = np.zeros((len(envelope), 1))
    mean_s1 = 0.122 * feature_fs
    std_s1 = 0.022 * feature_fs
    mean_s2 = 0.092 * feature_fs
    std_s2 = 0.022 * feature_fs

    for s in s1:
        upper_bound = np.round(np.min([len(states), s + mean_s1]))
        states[int(np.max([1, s])) - 1:int(upper_bound)] = 1

    for s in s2:
        lower_bound = int(np.max([s - np.floor(mean_s2 + std_s2), 1]))
        upper_bound = int(np.min([len(states), np.ceil(s + np.floor(mean_s2 + std_s2))]))
        search_window = np.multiply(envelope[lower_bound:upper_bound], 1 - states[lower_bound:upper_bound])
        s2_index = np.argmax(search_window)
        s2_index = int(np.min([len(states), lower_bound + s2_index - 1]))
        upper_bound = np.min([len(states), np.ceil(s2_index + (mean_s2 / 2))])
        states[int(np.max([np.ceil(s2_index - (mean_s2 / 2)), 0])):int(upper_bound)] = 3

        diffs = s1 - s
        pos_diffs = diffs[diffs >= 0]
        if len(pos_diffs):
            index_m = np.argmin(pos_diffs)
            end_pos = s1[index_m] - 1
        else:
            end_pos = len(states)
        states[int(np.ceil(s2_index + (mean_s2 / 2)) - 1):int(end_pos)] = 4

    def get_index_before(index):
        x, y = index
        if x == 0:
            return False
        return [x - 1, y]

    def get_index_after(index, h):
        x, y = index
        if x == h:
            return False
        return [x + 1, 0]

    first_location_of_definite_state = np.transpose(np.nonzero(states))[0]
    first_location_of_undefined_state = get_index_before(first_location_of_definite_state)

    if not first_location_of_undefined_state:
        print("no zeros")
    if states[first_location_of_definite_state[0], first_location_of_definite_state[1]] == 1:
        for i in range(first_location_of_undefined_state[0] + 1):
            states[i, 0] = 4
    elif states[first_location_of_definite_state[0], first_location_of_definite_state[1]] == 3:
        for i in range(first_location_of_undefined_state[0] + 1):
            states[i, 0] = 2

    last_location_of_definite_state = np.transpose(np.nonzero(states))[-1]
    last_location_of_undefined_state = get_index_after(last_location_of_definite_state, len(states))

    if not last_location_of_undefined_state:
        print("no zeros")
    elif states[last_location_of_definite_state[0], last_location_of_definite_state[1]] == 1:
        for i in range(last_location_of_undefined_state[0]):
            states[i, 0] = 2
    elif states[last_location_of_definite_state[0], last_location_of_definite_state[1]] == 3:
        for i in range(last_location_of_undefined_state[0], len(states)):
            states[i, 0] = 4

    states[states == 0] = 2
    return states


def get_duration(heart_rate, systolic_time, audio_seg_fs):
    mean_s1 = int(np.round(0.122 * audio_seg_fs))
    std_s1 = int(np.round(0.022 * audio_seg_fs))
    mean_s2 = int(np.round(0.094 * audio_seg_fs))
    std_s2 = int(np.round(0.022 * audio_seg_fs))

    mean_systole = int(np.round(systolic_time * audio_seg_fs)) - mean_s1
    std_systole = (25 / 1000) * audio_seg_fs

    mean_diastole = ((60 / heart_rate) - systolic_time - 0.094) * 50
    std_diastole = 0.07 * mean_diastole + (6 / 1000) * 50

    d_distributions = np.array([
        [mean_s1, std_s1 ** 2],
        [mean_systole, std_systole ** 2],
        [mean_s2, std_s2 ** 2],
        [mean_diastole, std_diastole ** 2]
    ])

    min_systole = mean_systole - 3 * (std_systole + std_s1)
    max_systole = mean_systole + 3 * (std_systole + std_s1)
    min_diastole = mean_diastole - 3 * std_diastole
    max_diastole = mean_diastole + 3 * std_diastole

    min_s1 = mean_s1 - 3 * std_s2
    if min_s1 < 1:
        min_s1 = 1
    min_s2 = mean_s2 - 3 * std_s2
    if min_s2 < 1:
        min_s2 = 1
    max_s1 = mean_s1 + 3 * std_s1
    max_s2 = mean_s2 + 3 * std_s2

    return {
        'd_distributions': d_distributions,
        'max_s1': max_s1, 'min_s1': min_s1,
        'max_s2': max_s2, 'min_s2': min_s2,
        'max_systole': max_systole, 'min_systole': min_systole,
        'max_diastole': max_diastole, 'min_diastole': min_diastole,
    }


def get_heart_rate(input_signal, audio_fs, apply_wavelet_denoise=True):
    input_signal = low_pass_filter(input_signal, 2, 400, audio_fs)
    input_signal = high_pass_filter(input_signal, 2, 25, audio_fs)
    input_signal = schmidt_spike_removal(input_signal, audio_fs)
    # 小波去噪：抑制持续性杂音，使后续包络峰值更准确
    if apply_wavelet_denoise:
        input_signal = wavelet_denoise(input_signal)

    hilbert_env = hilbert_envelope(input_signal, audio_fs)
    homomorphic_env = homomorphic_envelope(hilbert_env, audio_fs)

    # FFT-based autocorrelation: O(n log n) instead of O(n²)
    auto_correlation = _scipy_correlate(homomorphic_env, homomorphic_env, mode='full', method='fft')
    auto_correlation = auto_correlation[auto_correlation.size // 2:]

    min_index = int(0.5 * audio_fs) - 1
    max_index = int(2 * audio_fs) - 1

    index = np.argmax(auto_correlation[min_index:max_index])
    true_index = index + min_index - 1
    heart_rate = 60 / (true_index / audio_fs)

    max_sys_duration = int(np.round(((60 / heart_rate) * audio_fs) / 2)) - 1
    min_sys_duration = int(np.round(0.2 * audio_fs)) - 1

    pos = np.argmax(auto_correlation[min_sys_duration:max_sys_duration])
    systolic_time_interval = (min_sys_duration + pos) / audio_fs

    return {
        'heart_rate': heart_rate,
        'systolic_time_interval': systolic_time_interval,
    }


def get_pcg_features(audio_data, features_fs, audio_fs, wavelet=False,
                     apply_wavelet_denoise=True):
    audio_data = low_pass_filter(audio_data, 2, 400, audio_fs)
    audio_data = high_pass_filter(audio_data, 2, 25, audio_fs)
    audio_data = schmidt_spike_removal(audio_data, audio_fs)
    # 小波去噪：抑制持续性杂音后再提取包络特征，减少杂音对 HMM 发射概率的污染
    if apply_wavelet_denoise:
        audio_data = wavelet_denoise(audio_data)

    hilbert_env = hilbert_envelope(audio_data, audio_fs)
    normalized_hilbert = downsample(hilbert_env, features_fs, audio_fs)
    normalized_hilbert = normalize(normalized_hilbert)

    homomorphic_env = homomorphic_envelope(hilbert_env, audio_fs)
    normalized_homomorphic = downsample(homomorphic_env, features_fs, audio_fs)
    normalized_homomorphic = normalize(normalized_homomorphic)

    num_of_dims = 3 if wavelet else 2
    pcg_features = np.zeros((len(normalized_homomorphic), num_of_dims))
    pcg_features[:, 0] = normalized_homomorphic
    pcg_features[:, 1] = normalized_hilbert

    if wavelet:
        wv = wavelet_envelope(audio_data, audio_fs)
        wv = wv[1:len(normalized_homomorphic)]
        normalized_wavelet = downsample(wv, features_fs, audio_fs)
        normalized_wavelet = normalize(normalized_wavelet)
        pcg_features[:, 2] = normalized_wavelet

    return {'pcg_features': pcg_features, 'fs': features_fs}


def expand_qt(original_qt, old_fs, new_fs, new_length):
    original_qt = np.array(original_qt).T
    expanded_qt = np.zeros(new_length)
    indices_of_changes = np.where(np.abs(np.diff(original_qt)) > 0)
    indices_of_changes = np.append(indices_of_changes, len(original_qt))

    start_index = 0
    for i in range(len(indices_of_changes)):
        end_index = indices_of_changes[i]
        mid_point = round((end_index - start_index) / 2) + start_index
        value_at_mid_point = original_qt[mid_point]
        expanded_start_index = round(np.multiply(np.divide(start_index, old_fs), new_fs))
        expanded_end_index = round(np.multiply(np.divide(end_index, old_fs), new_fs))
        if expanded_end_index > new_length:
            expanded_qt[expanded_start_index:] = value_at_mid_point
        else:
            expanded_qt[expanded_start_index:expanded_end_index] = value_at_mid_point
        start_index = end_index

    return expanded_qt


def plot_segmentation(pred_state, recording, audio_fs, title='', save_path=None, show=True):
    """
    Plot PCG waveform (top) and color-coded segmentation states (bottom).
    States: 1=S1 (red), 2=Systole (orange), 3=S2 (green), 4=Diastole (blue)

    Parameters
    ----------
    pred_state : array-like  Predicted state sequence at audio_fs rate
    recording  : array-like  Raw PCG signal at audio_fs rate
    audio_fs   : int         Sampling rate (Hz)
    title      : str         Plot title (e.g. filename)
    save_path  : str or None Path to save PNG; None = do not save
    show       : bool        Whether to display the window
    """
    recording = np.array(recording).flatten()
    pred_state = np.array(pred_state).flatten()

    ts_si = np.linspace(0, (len(recording) - 1) / audio_fs, len(recording))
    ts_s = np.linspace(0, (len(pred_state) - 1) / audio_fs, len(pred_state))

    sig_norm = normalize(recording)

    fig, axes = plt.subplots(2, 1, figsize=(20, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # ---- Top: PCG waveform with shaded state regions ----
    axes[0].plot(ts_si, sig_norm, color='steelblue', linewidth=0.5, alpha=0.85)
    axes[0].set_ylabel('Amplitude (normalized)')
    axes[0].set_title('PCG Signal')
    axes[0].grid(True, alpha=0.3)

    changes = np.where(np.diff(pred_state) != 0)[0] + 1
    seg_starts = np.concatenate([[0], changes])
    seg_ends = np.concatenate([changes, [len(pred_state)]])

    for start, end in zip(seg_starts, seg_ends):
        state_val = pred_state[start]
        color = STATE_COLORS.get(state_val, 'gray')
        t0 = ts_s[int(start)]
        t1 = ts_s[min(int(end), len(ts_s) - 1)]
        axes[0].axvspan(t0, t1, alpha=0.18, color=color, linewidth=0)

    # ---- Bottom: Segmentation color bands ----
    for start, end in zip(seg_starts, seg_ends):
        state_val = pred_state[start]
        color = STATE_COLORS.get(state_val, 'gray')
        t0 = ts_s[int(start)]
        t1 = ts_s[min(int(end), len(ts_s) - 1)]
        axes[1].fill_between([t0, t1], [0, 0], [1, 1], color=color, alpha=0.8)

    axes[1].set_yticks([])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Segmentation States')
    axes[1].set_ylim(0, 1)

    patches = [mpatches.Patch(color=STATE_COLORS[k], label=STATE_LABELS[k])
               for k in [1, 2, 3, 4]]
    axes[1].legend(handles=patches, loc='upper right', ncol=4, fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
