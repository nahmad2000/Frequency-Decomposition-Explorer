import matplotlib.pyplot as plt
import numpy as np
import streamlit as st # Import Streamlit for plotting

# Configure Matplotlib for Streamlit (agg backend is usually good for non-interactive plots)
# plt.switch_backend('Agg') # Optional: sometimes needed depending on environment

def plot_signal(time, signal, title="Signal"):
    """
    Plots a 1D signal using Matplotlib and displays in Streamlit.

    Args:
        time (np.ndarray): Time axis data.
        signal (np.ndarray): Signal amplitude data.
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(time, signal)
    ax.set_title(title)
    ax.set_xlabel("Time (or Sample Index)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()
    return fig # Return the figure for st.pyplot

def plot_image(image, title="Image", cmap='gray'):
    """
    Displays a 2D image using Matplotlib and displays in Streamlit.

    Args:
        image (np.ndarray): 2D image data.
        title (str): Title for the plot.
        cmap (str): Colormap for the image (e.g., 'gray', 'viridis').

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis('off') # Hide axes for images
    # Add a colorbar if not grayscale or if desired
    # fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig # Return the figure for st.pyplot

def plot_spectrum(frequencies, magnitude_spectrum, title="Frequency Spectrum", is_1d=True):
    """
    Plots the magnitude spectrum (1D or 2D) using Matplotlib and displays in Streamlit.

    Args:
        frequencies (np.ndarray or tuple): Frequency axis/axes.
        magnitude_spectrum (np.ndarray): Magnitude of the FFT result (e.g., np.abs(fft_result)).
        title (str): Title for the plot.
        is_1d (bool): True for 1D spectrum, False for 2D.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object.
    """
    fig, ax = plt.subplots()

    if is_1d:
        # --- 1D Spectrum Plot ---
        ax.plot(frequencies, magnitude_spectrum)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Magnitude")
        # Optional: Use a log scale for magnitude if dynamic range is large
        # ax.set_yscale('log')
        ax.grid(True)
    else:
        # --- 2D Spectrum Plot ---
        # Display the log magnitude for better visualization of details
        log_spectrum = np.log1p(magnitude_spectrum) # Use log1p for stability (log(1+x))
        freq_rows, freq_cols = frequencies
        # Determine extent for imshow based on frequency vectors
        # Frequencies are shifted, so min/max represent the range
        extent = [freq_cols.min(), freq_cols.max(), freq_rows.min(), freq_rows.max()]
        im = ax.imshow(log_spectrum, cmap='viridis', extent=extent, aspect='auto', origin='lower')
        ax.set_xlabel("Frequency (cols)")
        ax.set_ylabel("Frequency (rows)")
        fig.colorbar(im, ax=ax, label='Log Magnitude')
        ax.axis('on') # Keep axes for 2D spectrum

    ax.set_title(title)
    plt.tight_layout()
    return fig # Return the figure for st.pyplot

def plot_reconstructions(original_data, bands_data, is_1d, band_titles):
    """
    Plots the original data and reconstructed bands in Streamlit.

    Args:
        original_data (np.ndarray): The original 1D signal or 2D image.
        bands_data (dict): Dictionary where keys are band names (str) and
                           values are the corresponding reconstructed np.ndarray.
        is_1d (bool): True if the data is 1D, False if 2D.
        band_titles (dict): Dictionary mapping band keys to display titles.

    Returns:
        matplotlib.figure.Figure: The Matplotlib figure object showing all plots.
    """
    num_bands = len(bands_data)
    # Total plots = original + number of bands
    num_plots = 1 + num_bands

    # Determine layout (e.g., vertical stack for 1D, grid for 2D)
    if is_1d:
        nrows = num_plots
        ncols = 1
        fig_height = 3 * num_plots # Adjust height based on number of plots
        fig_width = 6
    else:
        # Try to make a squarish grid
        ncols = int(np.ceil(np.sqrt(num_plots)))
        nrows = int(np.ceil(num_plots / ncols))
        fig_height = 4 * nrows
        fig_width = 4 * ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))
    # Flatten axes array for easy iteration, handle single plot case
    axes = np.array(axes).flatten()

    # --- Plot Original Data ---
    ax = axes[0]
    if is_1d:
        time_axis = np.arange(original_data.shape[0])
        ax.plot(time_axis, original_data)
        ax.set_ylabel("Amplitude")
        if nrows == 1 and ncols == 1: # Only one plot total (no bands?)
             ax.set_xlabel("Time (or Sample Index)")
    else:
        im = ax.imshow(original_data, cmap='gray')
        ax.axis('off')
        # fig.colorbar(im, ax=ax) # Optional colorbar
    ax.set_title(band_titles.get('Original', 'Original Data'))
    ax.grid(True if is_1d else False)

    # --- Plot Reconstructed Bands ---
    plot_index = 1
    for band_name, band_data in bands_data.items():
        if plot_index < len(axes): # Ensure we don't exceed subplot count
            ax = axes[plot_index]
            if is_1d:
                time_axis = np.arange(band_data.shape[0])
                ax.plot(time_axis, band_data)
                ax.set_ylabel("Amplitude")
                # Label x-axis only on the bottom-most plot in a column
                if (plot_index // ncols) == (nrows -1):
                     ax.set_xlabel("Time (or Sample Index)")
            else:
                im = ax.imshow(band_data, cmap='gray')
                ax.axis('off')
                # fig.colorbar(im, ax=ax) # Optional colorbar
            ax.set_title(band_titles.get(band_name, band_name))
            ax.grid(True if is_1d else False)
            plot_index += 1

    # Hide any unused subplots
    for i in range(plot_index, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(pad=1.5) # Add padding between subplots
    return fig # Return the figure for st.pyplot