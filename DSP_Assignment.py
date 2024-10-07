#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def low_pass_filter(signal, cutoff_freq, sample_rate):
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
    
    # Create a low-pass filter mask
    filter_mask = np.abs(fft_freq) < cutoff_freq
    
    # Apply the filter
    filtered_fft_signal = fft_signal * filter_mask
    
    # Perform inverse FFT
    filtered_signal = np.fft.ifft(filtered_fft_signal)
    
    return np.real(filtered_signal), fft_freq, fft_signal, filtered_fft_signal

def high_pass_filter(signal, cutoff_freq, sample_rate):
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sample_rate)
    
    # Create a high-pass filter mask
    filter_mask = np.abs(fft_freq) > cutoff_freq
    
    # Apply the filter
    filtered_fft_signal = fft_signal * filter_mask
    
    # Perform inverse FFT
    filtered_signal = np.fft.ifft(filtered_fft_signal)
    
    return np.real(filtered_signal), fft_freq, fft_signal, filtered_fft_signal

def band_pass_filter(signal, low_cutoff_freq, high_cutoff_freq, sample_rate):
    fft_signal = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1/sample_rate)

    # Create a band-pass filter mask
    filter_mask = (np.abs(fft_freq) > low_cutoff_freq) & (np.abs(fft_freq) < high_cutoff_freq)
    
    # Apply the filter
    filtered_fft_signal = fft_signal * filter_mask
    
    # Perform inverse FFT
    filtered_signal = np.fft.ifft(filtered_fft_signal)
    
    return np.real(filtered_signal), fft_freq, fft_signal, filtered_fft_signal

def plot_components(original_signal, filtered_signal, original_fft, filtered_fft, fft_freq, sample_rate, filter_type):
    time = np.arange(len(original_signal)) / sample_rate

    plt.figure(figsize=(14, 12))

    # Plot original signal
    plt.subplot(4, 1, 1)
    plt.plot(time, original_signal, label='Original Signal', color='blue')
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    # Plot filtered signal
    plt.subplot(4, 1, 2)
    plt.plot(time, filtered_signal, label=f'Filtered Signal ({filter_type})', color='orange')
    plt.title(f'Filtered Signal ({filter_type})')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.legend()

    # Plot original FFT
    plt.subplot(4, 1, 3)
    plt.plot(fft_freq, np.abs(original_fft), label='Original FFT', color='blue')
    plt.title('Original FFT')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0, np.max(fft_freq) // 2)  # Limit to positive frequencies
    plt.grid()
    plt.legend()

    # Plot filtered FFT
    plt.subplot(4, 1, 4)
    plt.plot(fft_freq, np.abs(filtered_fft), label=f'Filtered FFT ({filter_type})', color='orange')
    plt.title(f'Filtered FFT ({filter_type})')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.xlim(0, np.max(fft_freq) // 2)  # Limit to positive frequencies
    plt.grid()
    plt.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(f'filtered_plot_{filter_type}.png')
    plt.close()  # Close the plot to free up memory

def main(input_file, output_file, cutoff_freq, filter_type, high_cutoff_freq=None):
    # Read audio file
    sample_rate, data = wavfile.read(input_file)

    # Check if audio is stereo or mono
    if data.ndim == 2:
        # If stereo, apply filter to each channel
        filtered_data = np.zeros_like(data)
        for channel in range(data.shape[1]):
            if filter_type == 'low':
                filtered_data[:, channel], fft_freq, original_fft, filtered_fft = low_pass_filter(data[:, channel], cutoff_freq, sample_rate)
            elif filter_type == 'high':
                filtered_data[:, channel], fft_freq, original_fft, filtered_fft = high_pass_filter(data[:, channel], cutoff_freq, sample_rate)
            elif filter_type == 'band':
                filtered_data[:, channel], fft_freq, original_fft, filtered_fft = band_pass_filter(data[:, channel], cutoff_freq, high_cutoff_freq, sample_rate)
    else:
        # If mono, apply filter directly
        if filter_type == 'low':
            filtered_data, fft_freq, original_fft, filtered_fft = low_pass_filter(data, cutoff_freq, sample_rate)
        elif filter_type == 'high':
            filtered_data, fft_freq, original_fft, filtered_fft = high_pass_filter(data, cutoff_freq, sample_rate)
        elif filter_type == 'band':
            filtered_data, fft_freq, original_fft, filtered_fft = band_pass_filter(data, cutoff_freq, high_cutoff_freq, sample_rate)

    # Normalize the output data to avoid clipping
    filtered_data = np.int16(filtered_data / np.max(np.abs(filtered_data)) * 32767)

    # Write the filtered audio to a new file
    wavfile.write(output_file, sample_rate, filtered_data)

    print(f"Filtered audio saved to {output_file}")

    # Plot components and save as image
    plot_components(data, filtered_data, original_fft, filtered_fft, fft_freq, sample_rate, filter_type.capitalize())

if __name__ == "__main__":
    input_file = 'pianos.wav'  # Replace with your input file
    output_file = 'output.wav'  # Replace with your desired output file
    cutoff_freq = 500  # Cutoff frequency in Hz
    high_cutoff_freq = 1500  # High cutoff frequency for band-pass filter
    filter_type = 'band'  # Choose 'low', 'high', or 'band'

    main(input_file, output_file, cutoff_freq, filter_type, high_cutoff_freq)
