# Frequency Decomposition Explorer

![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://frequency-decomposition-explorer-nahmad.streamlit.app/)

A Streamlit application to interactively explore the frequency components of 1D signals (like ECG or synthetic waves) and 2D images. Upload your data, define frequency bands, and visualize how different frequencies contribute to the overall signal/image. You can also selectively reconstruct the data using only chosen frequency bands.

---
## 🚀 Live Demo

**➡️ [Try the Live Demo!](https://frequency-decomposition-explorer-nahmad.streamlit.app/)**

*(No installation required!)*

## 🖼️ Sample Outputs

![Demo 1](examples/demo1.png) 
![Demo 2](examples/demo2.png)

---
## ✨ Features

* **Upload Data:** Supports 1D signals (`.npy` arrays) and 2D images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`, 2D `.npy`).
* **Interactive Band Definition:** Define custom frequency bands (Low, Mid, High, etc.) directly in the sidebar UI. Frequencies are relative to the Nyquist frequency (0.0 to 0.5).
* **Frequency Analysis:** Computes the Fast Fourier Transform (FFT) to analyze the frequency content.
* **Visualization:**
    * Displays the original signal/image.
    * Shows the magnitude frequency spectrum (1D or 2D).
    * Plots the reconstructed signal/image corresponding to each defined frequency band individually.
* **Selective Reconstruction:** Choose which frequency bands to include in a final reconstruction via checkboxes in the sidebar.
* **Comparison:** View the original data side-by-side with the reconstruction built from only the selected frequency bands.
* **Difference Plot:** Visualize the information lost by displaying the difference between the original and the selectively reconstructed data.
* **Coefficient Reduction Metric:** Shows the percentage of frequency coefficients kept in the selective reconstruction, illustrating the potential for data reduction.

## 💻 Usage

### 1. Live Demo (Recommended)

The easiest way to use the application is via the hosted Streamlit Community Cloud app:

**➡️ [Try the Live Demo!](https://frequency-decomposition-explorer-nahmad.streamlit.app/)**

### 2. Local Execution

If you prefer to run the application locally:

1.  **Clone the Repository:**

```bash
git clone https://github.com/nahmad2000/Frequency-Decomposition-Explorer.git
cd Frequency-Decomposition-Explorer
```

2.  **Install Dependencies:**
    (Ensure you have Python 3.8 or newer installed)
```bash
pip install -r requirements.txt
```

3.  **Run the Streamlit App:**
```bash
streamlit run main.py
```

4.  The application should open automatically in your web browser (usually at `http://localhost:8501`).

## ⚙️ Installation (for Local Usage)

1.  **Clone:** Clone this repository to your local machine.
2.  **Python:** Ensure you have Python (version 3.8 or later recommended) and `pip` installed.
3.  **Dependencies:** Navigate to the project directory in your terminal and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    This installs libraries like Streamlit, NumPy, SciPy, Matplotlib, and Pillow.

---
## 🤔 How It Works

The application follows this workflow:

1.  **User Interface (Streamlit):** The `main.py` script builds the web interface using Streamlit. Users interact via the sidebar and main panel.
2.  **Data Upload:** The user uploads a 1D (`.npy`) or 2D (image file or `.npy`) via `st.file_uploader`.
3.  **Frequency Bands:** The user defines frequency bands (low/high cutoffs relative to Nyquist) and selects which bands to keep for final reconstruction using widgets in the sidebar (`main.py`).
4.  **FFT Computation:** The uploaded data is processed by `frequency_analyzer.py`'s `compute_fft` function to obtain the frequency domain representation (`fft_result`) and corresponding frequencies.
5.  **Band Filtering:** For each defined band:
    * A boolean mask is created using `get_band_mask` based on the band's frequency limits.
    * The original `fft_result` is multiplied by the mask using `apply_filter` to isolate the frequencies within that band. The result (`filtered_fft`) is stored.
6.  **Individual Band Visualization:** (Optional) The `filtered_fft` for each band is transformed back using `compute_ifft` and plotted using `plotter.py` functions via `st.pyplot` in `main.py`.
7.  **Selective Reconstruction:**
    * A combined mask is generated from the masks of only the user-selected ('kept') bands.
    * The original `fft_result` is filtered *once* using this combined mask.
    * The final reconstructed signal/image is obtained by applying `compute_ifft` to this combined filtered FFT data.
8.  **Display Results:** `main.py` uses `st.pyplot` and functions from `plotter.py` to display:
    * The original data.
    * The frequency spectrum.
    * The reconstruction from selected bands.
    * The difference between the original and the reconstruction.
    * The percentage of FFT coefficients retained.
