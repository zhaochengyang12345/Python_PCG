import numpy as np
from utils.tools import get_pcg_features, label_pcg_states, get_heart_rate, expand_qt
from utils.preprocessing import resample_to_fs
from models.viterbi_algorithm import viterbi_decode_pcg
from models.band_pi_matrices import train_band_pi_matrices


def train_segmentation_algorithm(train_recordings, train_annotations, features_fs, audio_fs, wavelet=False):
    number_of_states = 4
    num_pcgs = len(train_recordings)
    state_observation_values = []

    for pcgi in range(num_pcgs):
        state_observation_values.append([0, 0, 0, 0])
        pcg_audio = np.squeeze(train_recordings[pcgi])

        s1_locations = train_annotations[pcgi, 0].flatten()
        s2_locations = train_annotations[pcgi, 1].flatten()

        pcg_features = get_pcg_features(pcg_audio, features_fs, audio_fs, wavelet)['pcg_features']
        pcg_states = label_pcg_states(pcg_features[:, 0], s1_locations, s2_locations, pcg_audio, features_fs)
        for state_i in range(number_of_states):
            state_observation_values[pcgi][state_i] = pcg_features[np.where(pcg_states == state_i + 1)[0], :]

    bpm = train_band_pi_matrices(state_observation_values)
    return {
        'total_obs_distribution': bpm['total_obs_distribution'],
        'pi_vector': bpm['pi_vector'],
        'model': bpm['model'],
    }


def predict_segmentation(recording, features_fs, audio_fs, pi_vector, model,
                         total_observation_distribution, processing_fs=None):
    """
    Run segmentation on a PCG recording without ground-truth annotations.

    Parameters
    ----------
    processing_fs : int or None
        Internally resample the signal to this rate before processing.
        Use 1000 for best speed (algorithm was trained at 1000 Hz).
        Defaults to audio_fs (no resampling).

    Returns
    -------
    pred_state : ndarray, shape (len(recording),)
        Predicted state at every sample (1=S1, 2=Systole, 3=S2, 4=Diastole)
    """
    recording = np.array(recording).flatten()
    orig_len = len(recording)

    # Resample to processing_fs for efficiency
    proc_fs = int(processing_fs) if processing_fs else audio_fs
    if proc_fs != audio_fs:
        recording_proc = resample_to_fs(recording, audio_fs, proc_fs)
    else:
        recording_proc = recording

    pcg_features = get_pcg_features(recording_proc, features_fs, proc_fs)['pcg_features']
    heart_rate_info = get_heart_rate(recording_proc, proc_fs)
    _, _, qt = viterbi_decode_pcg(
        pcg_features, pi_vector, model, total_observation_distribution,
        heart_rate_info['heart_rate'], heart_rate_info['systolic_time_interval'],
        features_fs,
    )
    # Expand qt back to original audio_fs timeline
    return expand_qt(qt, features_fs, audio_fs, orig_len)
