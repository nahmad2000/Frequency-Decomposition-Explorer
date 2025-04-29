import numpy as np
from numpy.fft import fft, fft2, fftfreq, fftshift, ifft, ifft2, ifftshift

def compute_fft(data):
    """
    Computes the Fast Fourier Transform (FFT) of the input data.

    Args:
        data (np.ndarray): Input data, 1D (signal) or 2D (image).

    Returns:
        tuple: A tuple containing:
            - fft_result (np.ndarray): Complex array representing the frequency domain (fftshifted).
            - frequencies (tuple or np.ndarray): Frequency axes corresponding to fft_result.
              For 1D, it's a single array. For 2D, it's a tuple of two arrays (freq_x, freq_y).
    """
    try:
        if data.ndim == 1:
            # --- 1D FFT ---
            # Compute FFT
            fft_result = fft(data)
            # Compute corresponding frequencies
            n = data.size
            # Sample spacing (assume unit time or pixel spacing if not provided)
            sample_spacing = 1.0
            frequencies = fftfreq(n, d=sample_spacing)
            # Shift zero frequency component to the center
            fft_result = fftshift(fft_result)
            frequencies = fftshift(frequencies)
            return fft_result, frequencies
        elif data.ndim == 2:
            # --- 2D FFT ---
            # Compute 2D FFT
            fft_result = fft2(data)
            # Compute corresponding frequency grids
            rows, cols = data.shape
            # Sample spacing (assume unit pixel spacing if not provided)
            sample_spacing_row = 1.0
            sample_spacing_col = 1.0
            freq_rows = fftfreq(rows, d=sample_spacing_row)
            freq_cols = fftfreq(cols, d=sample_spacing_col)
            # Shift zero frequency component to the center
            fft_result = fftshift(fft_result)
            freq_rows = fftshift(freq_rows)
            freq_cols = fftshift(freq_cols)
            # Return shifted FFT and frequency axes as a tuple
            return fft_result, (freq_rows, freq_cols)
        else:
            raise ValueError("Input data must be 1D or 2D.")
    except Exception as e:
        print(f"Error during FFT computation: {e}")
        # Return None or raise a custom exception if preferred
        return None, None

def get_band_mask(frequencies, fft_shape, freq_low, freq_high, is_1d):
    """
    Creates a boolean mask for a specific frequency band.

    Args:
        frequencies (tuple or np.ndarray): Frequency axes from compute_fft.
        fft_shape (tuple): Shape of the FFT result array.
        freq_low (float or None): Lower cutoff frequency. None means no lower bound.
        freq_high (float or None): Upper cutoff frequency. None means no upper bound.
        is_1d (bool): True if the data is 1D, False if 2D.

    Returns:
        np.ndarray: Boolean mask with the same shape as the FFT result.
    """
    mask = np.zeros(fft_shape, dtype=bool) # Start with all False

    if is_1d:
        # --- 1D Mask ---
        if freq_low is None and freq_high is None:
            mask[:] = True # Keep all frequencies if no bounds
        elif freq_low is None:
             mask[frequencies <= freq_high] = True
        elif freq_high is None:
             mask[frequencies >= freq_low] = True
        else:
             mask[(frequencies >= freq_low) & (frequencies <= freq_high)] = True
    else:
        # --- 2D Mask (using radial frequency) ---
        freq_rows, freq_cols = frequencies
        # Create coordinate grids from frequency vectors
        center_row, center_col = fft_shape[0] // 2, fft_shape[1] // 2
        col_indices, row_indices = np.meshgrid(np.arange(fft_shape[1]), np.arange(fft_shape[0]))

        # Calculate radial distance (frequency magnitude) from the center (zero frequency)
        # Use the actual frequency values for scaling
        scale_row = freq_rows[-1] - freq_rows[0] if len(freq_rows) > 1 else 1
        scale_col = freq_cols[-1] - freq_cols[0] if len(freq_cols) > 1 else 1

        # More robust way to calculate radial frequency using freq vectors
        # Ensure broadcasting works correctly
        fx = freq_cols[np.newaxis, :] # Shape (1, N_cols)
        fy = freq_rows[:, np.newaxis] # Shape (N_rows, 1)
        radial_freq = np.sqrt(fx**2 + fy**2) # Shape (N_rows, N_cols)

        # Apply frequency thresholds
        if freq_low is None and freq_high is None:
            mask[:] = True # Keep all frequencies
        elif freq_low is None:
            mask[radial_freq <= freq_high] = True
        elif freq_high is None:
            mask[radial_freq >= freq_low] = True
        else:
            mask[(radial_freq >= freq_low) & (radial_freq <= freq_high)] = True

    return mask

def apply_filter(fft_result, mask):
    """
    Applies a frequency mask to the FFT result.

    Args:
        fft_result (np.ndarray): Complex FFT array (output of compute_fft).
        mask (np.ndarray): Boolean mask (output of get_band_mask).

    Returns:
        np.ndarray: Filtered complex FFT array.
    """
    # Ensure mask is boolean for multiplication (True=1, False=0)
    filtered_fft = fft_result * mask.astype(complex)
    return filtered_fft

def compute_ifft(filtered_fft_result, is_1d):
    """
    Computes the Inverse Fast Fourier Transform (IFFT).

    Args:
        filtered_fft_result (np.ndarray): Filtered complex FFT array.
        is_1d (bool): True if the original data was 1D, False if 2D.

    Returns:
        np.ndarray: Reconstructed real data (signal or image).
    """
    try:
        # Shift back the zero frequency component before IFFT
        shifted_back_fft = ifftshift(filtered_fft_result)

        if is_1d:
            # --- 1D IFFT ---
            reconstructed_data = ifft(shifted_back_fft)
        else:
            # --- 2D IFFT ---
            reconstructed_data = ifft2(shifted_back_fft)

        # Return the real part (imaginary part should be negligible for real inputs)
        return np.real(reconstructed_data)
    except Exception as e:
        print(f"Error during IFFT computation: {e}")
        # Return None or raise a custom exception
        return None