import librosa
import numpy as np


def extract_mfcc(
    file_path,
    sample_rate=22050,
    n_mfcc=40,
    max_pad_len=174
):
    """
    Extract MFCC features using the same preprocessing
    pipeline used during model training.
    """

    audio, sr = librosa.load(
        file_path,
        sr=sample_rate
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # Pad or truncate to exactly 174 time frames
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :max_pad_len]

    # Per-sample normalization
    mean = mfcc.mean()
    std = mfcc.std()

    mfcc = (mfcc - mean) / (std + 1e-8)

    # Add CNN channel dimension
    mfcc = mfcc[..., np.newaxis]

    # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc