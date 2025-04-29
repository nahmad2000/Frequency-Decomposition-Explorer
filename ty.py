import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt # Keep for plotting

# --- Parameters for the signal ---
sample_rate = 1000  # Samples per second (Hz) - Common for ECG analysis
duration = 10     # Seconds - Let's make it a bit longer for ECG
heart_rate = 75   # Average heart rate in beats per minute
noise_amplitude = 0.05 # Adjust noise level as needed (0 for clean ECG)

# --- Generate Synthetic ECG Signal using NeuroKit2 ---
print(f"Generating {duration}s ECG signal at {sample_rate} Hz (HR: {heart_rate} bpm)...")
try:
    ecg_signal = nk.ecg_simulate(duration=duration,
                                 sampling_rate=sample_rate,
                                 heart_rate=heart_rate,
                                 noise = 0) # Generate clean ECG first
    print("Clean ECG signal generated successfully.")
except Exception as e:
    print(f"Error generating ECG signal: {e}")
    ecg_signal = None

if ecg_signal is not None:
    # --- Add Noise ---
    if noise_amplitude > 0:
        print(f"Adding noise (amplitude: {noise_amplitude})...")
        # Generate noise with the same length as the signal
        num_samples = len(ecg_signal)
        noise = noise_amplitude * np.random.randn(num_samples)
        signal_with_noise = ecg_signal + noise
    else:
        print("No noise added.")
        signal_with_noise = ecg_signal # Use the clean signal

    # --- Save the final signal as a .npy file ---
    output_filename = 'sample_ecg_signal.npy'
    np.save(output_filename, signal_with_noise)
    print(f"Saved final signal (ECG + Noise) to '{output_filename}'")

    # --- Optional: Plotting ---
    print("Plotting the generated signal...")
    time = np.arange(len(signal_with_noise)) / sample_rate # Create time axis

    plt.figure(figsize=(15, 5)) # Wider plot for ECG
    plt.plot(time, signal_with_noise, label='ECG + Noise')
    if noise_amplitude > 0:
        # Optionally plot the clean ECG for comparison if noise was added
        plt.plot(time, ecg_signal, label='Clean ECG', alpha=0.5, linestyle='--')
        plt.legend()

    plt.title(f"Generated ECG Signal (HR: {heart_rate} bpm, Noise: {noise_amplitude})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (arbitrary units)")
    plt.grid(True)
    plt.show()

else:
    print("Could not generate or save the signal.")