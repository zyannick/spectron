import pytest
import numpy as np


def test_add_gaussian_noise():
    """Test the add_gaussian_noise function"""
    from utils.data_augmentations import add_noise_according_to_db
    original_signal = np.random.rand(1000)
    target_snr_db = 80
    noisy_signal = add_noise_according_to_db(original_signal=original_signal, target_snr_db=target_snr_db)
    assert noisy_signal.shape == original_signal.shape





