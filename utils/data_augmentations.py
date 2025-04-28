import numpy as np


# Function to apply wavelength shift
def apply_wavelength_shift(wavelength,  shift_amount):
    shifted_wavelength = wavelength + shift_amount
    return shifted_wavelength



def add_noise_according_to_db(original_signal: np.ndarray, target_snr_db: float = 20):
    """
    Add noise to a signal
    Args:
        original_signal : np.ndarray the input signal
        target_snr_db : float the amount of noise to add
    Output:
        noisy_signal: np.ndarray noisy signal
    """
    signal_power = original_signal**2
    # Set a target SNR
    target_snr_db = target_snr_db
    # Calculate signal power and convert to dB
    sig_avg_signal = np.mean(signal_power)
    sig_avg_db = 10 * np.log10(sig_avg_signal)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_signal = np.random.normal(
        mean_noise, np.sqrt(noise_avg_watts), signal_power.shape  # type: ignore
    )
    # Noise up the original signal
    noisy_signal = original_signal + noise_signal
    return noisy_signal


class Composer(object):
    """
    Compose the list of transformations
    """

    name = "composer"

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transformation(self, transformation: object):
        """Add a new transformation

        Args:
            transformation (object): a transformer
        """
        if self.transforms is None:
            self.transforms = []
        if transformation is not None:
            self.transforms.append(transformation)

    def __call__(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """Call function of the augmentation"""
        for t in self.transforms:
            x = t(x)
        return x


class Gaussian_Noise(object):
    """Add poisson noise to the image"""

    name = "aug_gaussian_noise"

    def __init__(self, chance=0.5):
        self.chance = chance

    def __call__(
        self, x: np.ndarray) -> np.ndarray:
        """Call function of the augmentation"""
        if np.random.rand() < self.chance:
            x_noise = add_noise_according_to_db(original_signal=x, target_snr_db=np.random.randint(1, 50))
            x = np.copy(x_noise)

        return x
    


class Poisson_Noise(object):
    """Add poisson noise to the image"""

    name = "aug_posson_noise"

    def __init__(self, chance=0.5):
        self.chance = chance

    def __call__(
        self, x: np.ndarray) -> np.ndarray:
        """Call function of the augmentation"""
        if np.random.rand() < self.chance:

            vals = len(np.unique(x))
            vals = 2 ** np.ceil(np.log2(vals))
            x = (np.random.poisson(x * vals) / float(vals)).astype(np.uint8)

        return x


class Wavelength_Shift(object):
    """Add poisson noise to the image"""

    name = "aug_wavelength_shift"

    def __init__(self, chance=0.5):
        self.chance = chance

    def __call__(
        self, x: np.ndarray) -> np.ndarray:
        """Call function of the augmentation"""
        if np.random.rand() < self.chance:
            shift_amount = np.random.randint(-10, 10)
            x = apply_wavelength_shift(wavelength=x, shift_amount=shift_amount)

        return x


