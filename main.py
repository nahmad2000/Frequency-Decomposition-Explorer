import streamlit as st
import numpy as np
from PIL import Image
import io

# Import functions from other modules
import frequency_analyzer as fa
import plotter

# --- Helper Function to Load Data ---
# (Keep the previously corrected load_data_from_upload function here)
def load_data_from_upload(uploaded_file):
    """
    Loads data from a Streamlit UploadedFile object.
    Handles common image types and .npy files.

    Args:
        uploaded_file (streamlit.uploaded_file_manager.UploadedFile):
            The file uploaded via st.file_uploader.

    Returns:
        np.ndarray: The loaded data as a NumPy array, or None if loading fails.
        bool: True if data is 1D, False if 2D, None on failure.
        str: Error message if loading fails, else None.
    """
    if uploaded_file is None:
        return None, None, "No file uploaded."

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()

    try:
        # --- FIX: Initialize error_msg to None ---
        error_msg = None
        data = None
        is_1d = None
        # -----------------------------------------

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            # --- Load Image ---
            image = Image.open(io.BytesIO(file_bytes))
            data = np.array(image.convert('L')) # 'L' mode for grayscale
            is_1d = False
            st.success(f"Loaded image: {filename} (shape: {data.shape})")

        elif filename.lower().endswith('.npy'):
            # --- Load NumPy array ---
            data_loaded = np.load(io.BytesIO(file_bytes))
            if data_loaded.ndim == 1:
                 data = data_loaded # Assign to data variable
                 is_1d = True
                 st.success(f"Loaded 1D signal: {filename} (length: {data.shape[0]})")
            elif data_loaded.ndim == 2:
                 data = data_loaded # Assign to data variable
                 is_1d = False
                 st.success(f"Loaded 2D data from .npy: {filename} (shape: {data.shape})")
            else:
                 # Error case within .npy handling
                 is_1d = None
                 data = None # Ensure data is None on error
                 error_msg = f"Unsupported array dimension in {filename}: {data_loaded.ndim}D. Only 1D or 2D supported."
                 st.error(error_msg)

        else:
            # Unsupported file type case
            data = None # Ensure data is None
            is_1d = None # Ensure is_1d is None
            error_msg = f"Unsupported file type: {filename}. Please upload an image (PNG, JPG, TIF, BMP) or a 1D/2D NumPy array (.npy)."
            st.warning(error_msg)

        # Return the results - error_msg will be None on success, or a string on handled errors
        return data, is_1d, error_msg

    except Exception as e:
        # Catch any other unexpected exceptions during loading/processing
        error_msg = f"Error loading file {filename}: {e}"
        st.error(error_msg)
        return None, None, error_msg


# --- Streamlit App ---
def run_app():
    """Main function to run the Streamlit application."""

    st.set_page_config(layout="wide")
    st.title("📊 Frequency Decomposition & Reconstruction Explorer")

    st.markdown("""
    Upload a 1D signal or 2D image, define frequency bands, and visualize the components.
    **New:** Select bands in the sidebar to reconstruct the signal using only their information,
    demonstrating the principle of frequency-based data reduction.
    """)

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("⚙️ Controls")

        uploaded_file = st.file_uploader("Choose a file (.npy, .png, .jpg, .tif, .bmp)",
                                         type=['npy', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])

        st.subheader("Frequency Bands")
        st.markdown("Define bands (relative freq: 0.0 to 0.5). Use `None` for no upper bound.")

        # --- Band Management (using Session State) ---
        if 'bands' not in st.session_state:
            # Initialize with the new default 8 bands
            st.session_state.bands = {
                'Band 1 (0.005-0.010)': {'low': 0.005, 'high': 0.010, 'keep': True}, # Add 'keep' state
                'Band 2 (0.010-0.015)': {'low': 0.010, 'high': 0.015, 'keep': True},
                'Band 3 (0.015-0.020)': {'low': 0.015, 'high': 0.020, 'keep': True},
                'Band 4 (0.020-0.025)': {'low': 0.020, 'high': 0.025, 'keep': True},
                'Band 5 (0.025-0.050)': {'low': 0.025, 'high': 0.050, 'keep': True},
                'Band 6 (0.050-0.100)': {'low': 0.050, 'high': 0.100, 'keep': True},
                'Band 7 (0.100-0.200)': {'low': 0.100, 'high': 0.200, 'keep': True},
                'Band 8 (0.200-0.500)': {'low': 0.200, 'high': 0.500, 'keep': True}
            }
            st.session_state.next_band_id = 8

        # --- Display Editable Bands and Add Checkboxes ---
        bands_to_remove = []
        bands_new_entry = {}
        st.markdown("**Select Bands to Keep for Reconstruction:**")
        for band_name, band_info in st.session_state.bands.items():
            col1, col2, col3, col4 = st.columns([1, 3, 3, 1]) # Added column for checkbox
            with col1:
                 # Checkbox to select band for reconstruction
                 keep_band = st.checkbox("", value=band_info.get('keep', True), key=f"keep_{band_name}", help=f"Include '{band_name}' in final reconstruction")
                 bands_new_entry[band_name] = {'keep': keep_band} # Store selection state
            with col2:
                 # Keep existing low freq input
                 new_low = st.number_input(f"{band_name} - Low",
                                           value=band_info['low'] if band_info['low'] is not None else 0.0,
                                           min_value=0.0, max_value=0.5, step=0.005, format="%.3f", key=f"low_{band_name}")
                 bands_new_entry[band_name]['low'] = new_low
            with col3:
                 # Keep existing high freq input (with None handling)
                 current_high_val = band_info['high'] if band_info['high'] is not None else 0.5
                 is_high_none = band_info['high'] is None
                 use_upper_bound = st.checkbox("Set High", value=not is_high_none, key=f"use_high_{band_name}")

                 if use_upper_bound:
                     new_high = st.number_input(f"High",
                                                 value=current_high_val,
                                                 min_value=0.0, max_value=0.5, step=0.005, format="%.3f", key=f"high_{band_name}")
                     bands_new_entry[band_name]['high'] = new_high
                 else:
                     bands_new_entry[band_name]['high'] = None # Set to None if checkbox unchecked
            with col4:
                 # Keep delete button
                 if st.button("❌", key=f"del_{band_name}", help=f"Remove '{band_name}' band"):
                    bands_to_remove.append(band_name)

        # Apply updates and removals
        for band_name in bands_to_remove:
            del st.session_state.bands[band_name]
        # Update existing bands with potentially changed values (low, high, keep)
        for band_name, new_values in bands_new_entry.items():
            if band_name in st.session_state.bands: # Ensure band still exists
                st.session_state.bands[band_name].update(new_values)

        # --- Add New Band Logic (Unchanged) ---
        new_band_name = st.text_input("New Band Name", key="new_band_name_input")
        if st.button("➕ Add Band", key="add_band_button"):
            if new_band_name and new_band_name not in st.session_state.bands:
                # Add 'keep': True for new bands by default
                st.session_state.bands[new_band_name] = {'low': 0.0, 'high': 0.1, 'keep': True}
                st.rerun()
            elif not new_band_name:
                st.warning("Please enter a name for the new band.")
            else:
                st.warning(f"Band name '{new_band_name}' already exists.")

        # Analysis Button
        analyze_button = st.button("🚀 Analyze and Reconstruct", type="primary", disabled=(uploaded_file is None))

    # --- Main Area for Results ---
    if analyze_button and uploaded_file is not None:
        st.header("📊 Results")
        data, is_1d, error_msg = load_data_from_upload(uploaded_file)

        if data is not None and is_1d is not None:
            progress_bar = st.progress(0, text="Starting analysis...")
            try:
                # 1. Compute FFT
                progress_bar.progress(10, text="Computing FFT...")
                fft_result, frequencies = fa.compute_fft(data)
                if fft_result is None:
                    st.error("FFT computation failed.")
                    st.stop()
                magnitude_spectrum = np.abs(fft_result)

                # 2. Plot Original Data and Spectrum
                progress_bar.progress(20, text="Plotting original data & spectrum...")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Data")
                    if is_1d:
                        time_axis = np.arange(data.shape[0])
                        fig_orig = plotter.plot_signal(time_axis, data, title=f"Original Signal ({uploaded_file.name})")
                    else:
                        fig_orig = plotter.plot_image(data, title=f"Original Image ({uploaded_file.name})")
                    st.pyplot(fig_orig)
                with col2:
                    st.subheader("Frequency Spectrum")
                    if is_1d: display_freq = frequencies
                    else: display_freq = frequencies # plotter handles tuple
                    fig_spec = plotter.plot_spectrum(display_freq, magnitude_spectrum, title="Magnitude Spectrum", is_1d=is_1d)
                    st.pyplot(fig_spec)

                # 3. Process Bands and Prepare for Combined Reconstruction
                progress_bar.progress(40, text="Filtering frequency bands...")
                st.subheader("Individual Band Contributions")
                if not st.session_state.bands:
                     st.warning("No frequency bands defined.")
                     progress_bar.progress(100, text="Analysis complete (no bands defined).")
                     st.stop()

                band_filtered_ffts = {} # Store complex filtered FFT for each band
                reconstructed_bands_individual = {} # Store individual reconstructions for plotting
                combined_mask = np.zeros_like(fft_result, dtype=bool) # To track kept coefficients
                total_bands = len(st.session_state.bands)
                current_band_num = 0

                # Use columns for individual band plots
                num_cols = 3 # Adjust as needed
                cols_bands = st.columns(num_cols)
                col_idx = 0

                for band_name, band_info in st.session_state.bands.items():
                    current_band_num += 1
                    progress = 40 + int(40 * (current_band_num / total_bands))
                    progress_bar.progress(progress, text=f"Processing band: {band_name}...")

                    freq_low_abs = band_info['low']
                    freq_high_abs = band_info['high']

                    mask = fa.get_band_mask(frequencies, fft_result.shape, freq_low_abs, freq_high_abs, is_1d)
                    filtered_fft = fa.apply_filter(fft_result, mask)
                    band_filtered_ffts[band_name] = filtered_fft # Store for potential combination

                    # Reconstruct individually for visualization
                    reconstructed_data = fa.compute_ifft(filtered_fft, is_1d)
                    if reconstructed_data is not None:
                        reconstructed_bands_individual[band_name] = reconstructed_data
                        # Plot individual band contribution
                        with cols_bands[col_idx % num_cols]:
                            band_title = f"{band_name}"
                            if is_1d:
                                fig_band = plotter.plot_signal(np.arange(reconstructed_data.shape[0]), reconstructed_data, title=band_title)
                            else:
                                fig_band = plotter.plot_image(reconstructed_data, title=band_title)
                            st.pyplot(fig_band)
                            col_idx += 1

                    # Track mask for selected bands
                    if band_info.get('keep', False): # Check the 'keep' status
                        combined_mask = combined_mask | mask # Combine masks for selected bands

                # 4. Create Combined Reconstruction based on selected bands
                progress_bar.progress(85, text="Combining selected bands...")
                st.subheader("Reconstruction from Selected Bands")

                # Apply the combined mask from selected bands
                combined_filtered_fft = fa.apply_filter(fft_result, combined_mask)

                # Compute IFFT *once* on the combined filtered data
                reconstructed_combined = fa.compute_ifft(combined_filtered_fft, is_1d)

                if reconstructed_combined is not None:
                    # Calculate difference
                    difference = data - reconstructed_combined

                    # Calculate 'compression' metric
                    total_coeffs = fft_result.size
                    kept_coeffs = np.sum(combined_mask) # Number of True values in the combined mask
                    kept_percentage = (kept_coeffs / total_coeffs) * 100 if total_coeffs > 0 else 0

                    st.markdown(f"**Kept {kept_coeffs} out of {total_coeffs} frequency coefficients ({kept_percentage:.2f}%)**")
                    st.markdown("_(Note: This demonstrates coefficient reduction. Actual file size depends on quantization & entropy coding.)_")


                    # Plot Original vs Combined Reconstruction vs Difference
                    col1_rec, col2_rec = st.columns(2)
                    with col1_rec:
                        st.write("Original")
                        if is_1d:
                             fig_orig_comp = plotter.plot_signal(np.arange(data.shape[0]), data, title="Original")
                        else:
                             fig_orig_comp = plotter.plot_image(data, title="Original")
                        st.pyplot(fig_orig_comp)

                    with col2_rec:
                        st.write("Reconstructed (Selected Bands)")
                        if is_1d:
                             fig_rec_comp = plotter.plot_signal(np.arange(reconstructed_combined.shape[0]), reconstructed_combined, title="Reconstructed")
                        else:
                             fig_rec_comp = plotter.plot_image(reconstructed_combined, title="Reconstructed")
                        st.pyplot(fig_rec_comp)

                    # Optional: Plot Difference
                    st.subheader("Difference (Original - Reconstructed)")
                    if is_1d:
                        fig_diff = plotter.plot_signal(np.arange(difference.shape[0]), difference, title="Difference Signal")
                    else:
                        # Scale difference for visibility if needed, add a divergent colormap
                        diff_display = difference - np.mean(difference) # Center around zero
                        vmax = np.max(np.abs(diff_display))
                        fig_diff = plotter.plot_image(diff_display, title="Difference Image", cmap='coolwarm')#, vmin=-vmax, vmax=vmax)

                    st.pyplot(fig_diff)

                else:
                    st.error("Failed to create combined reconstruction.")

                progress_bar.progress(100, text="Analysis Complete!")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                import traceback
                st.error(traceback.format_exc()) # More detailed error for debugging
                progress_bar.progress(100, text="Analysis failed.")

        elif error_msg:
            pass # Error already shown


# --- Entry Point ---
if __name__ == "__main__":
    run_app()