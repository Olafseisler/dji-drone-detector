import SoapySDR
import numpy as np
from numpy import typing
import matplotlib.pyplot as plt
from scipy.signal import correlate, find_peaks, peak_prominences
from SoapySDR import *
import time
import sys

SYMBOL_TIME_S = 0.000072 # Default symbol time for OcuSync
SAMPLE_RATE = 20e6
CENTER_FREQ = 2450e6
CORR_BLOCK_LEN_SAMPLES = 256
SYMBOL_LEN_SAMPLES = 2 * int(SYMBOL_TIME_S * SAMPLE_RATE)
SLIDING_WINDOW_SIZE = SYMBOL_LEN_SAMPLES * 36


def get_signal_on_current(corr_results, symbol_len_samples, tolerance=0.2):
    """
    Get the signal state based on correlation of current data block
    :param corr_results: The correlation results as a real numpy array
    :param symbol_len_samples: The length of a symbol in samples
    :param tolerance: The tolerance for the mean peak distance
    :return: True if the signal is on, False otherwise
    """
    lower_bound = np.nanpercentile(corr_results, 99.9)

    peaks, _ = find_peaks(corr_results, height=lower_bound, distance=SYMBOL_LEN_SAMPLES * 0.8)
    # plot_correlation_results(corr_results, peaks, lower_bound, name="correlation_results") # For debugging
    if len(peaks) < int(0.2 * len(corr_results) / symbol_len_samples):
        return False

    # Filter out peaks too far apart
    diffs = np.diff(peaks)
    diffs = diffs[diffs < 3 * symbol_len_samples]
    mean_peak_dist = np.mean(diffs)
    signal_state = (1 - tolerance) * symbol_len_samples < mean_peak_dist < (1 + tolerance) * symbol_len_samples

    # if signal_state:
    #     print("Mean peak distance:", mean_peak_dist)
    #     # Write corr results
    #     with open("recordings/corr_data_interactive.txt", "w") as f:
    #         for j in corr_results:
    #             f.write(str(j) + "\n")
    #     plot_correlation_results(corr_results, peaks, lower_bound, name="correlation_results")

    return signal_state, diffs


def get_correlation(complex_signal):
    """
    Correlate two blocks shifted by a symbol's length. Return the square magnitude of the correlation
    :param complex_signal: Complex64 numpy array
    :return: The correlation results as a real numpy array
    """

    complex_signal -= np.mean(complex_signal)
    symbol_len_samples = int(SAMPLE_RATE * 0.000072)  # 72 microseconds
    shift = 150  # Estimated prefix length
    steps = range(0, len(complex_signal) - symbol_len_samples, CORR_BLOCK_LEN_SAMPLES // 2)

    num_steps = len(steps)
    corr_len = CORR_BLOCK_LEN_SAMPLES
    corr_results = np.empty(num_steps * corr_len, dtype=np.int32)

    for i, start in enumerate(steps):
        end1 = start + CORR_BLOCK_LEN_SAMPLES
        start2 = start + symbol_len_samples - shift
        end2 = start2 + CORR_BLOCK_LEN_SAMPLES if start2 + CORR_BLOCK_LEN_SAMPLES < len(complex_signal) else len(
            complex_signal)
        block_1 = complex_signal[start:end1]
        block_2 = complex_signal[start2:end2]

        # Pad the blocks with zeros if they are not the same length
        block_2 = np.pad(block_2, (0, len(block_1) - len(block_2)))

        corr = correlate(block_1, block_2, mode='same', method='direct')
        corr_results[i * corr_len:(i + 1) * corr_len] = np.abs(corr)

    return corr_results


def plot_correlation_results(corr_results, peaks, lower_bound, name="correlation_results"):
    plt.plot(corr_results, color='b')
    plt.axhline(np.mean(corr_results), color='r', linestyle='--')
    plt.axhline(lower_bound, color='g', linestyle='--')
    iqr = np.percentile(corr_results, 75) - np.percentile(corr_results, 25)
    tukey_limit = np.percentile(corr_results, 75) + 1.5 * iqr
    plt.axhline(tukey_limit, color='m', linestyle='--')
    plt.scatter(peaks, corr_results[peaks], color='r')

    # # Plot the peak prominences
    # prominences = peak_prominences(corr_results, peaks, wlen=400)[0]
    # contour_heights = corr_results[peaks] - prominences
    # plt.vlines(x=peaks, ymin=contour_heights, ymax=corr_results[peaks], color='r')

    plt.title(name)
    plt.show()


def add_noise_to_signal(signal, snr):
    """
    Add noise to a signal to achieve a desired SNR
    :param signal: The complex signal to add noise to
    :param snr: The desired signal-to-noise ratio in dB
    :return: The noisy signal
    """
    linear_snr = 10 ** (snr / 10)
    avg_signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = avg_signal_power / linear_snr

    noise_real = np.random.normal(0, 1, len(signal))
    noise_imag = np.random.normal(0, 1, len(signal))
    noise = noise_real + 1j * noise_imag
    noise *= np.sqrt(noise_power / np.mean(np.abs(noise) ** 2))

    return signal + noise


def stream_from_sdr(center_freq, sample_rate, duration):
    """
    Stream from an SDR and detect a signal
    :param center_freq: Center frequency in Hz
    :param sample_rate: Sample rate in Hz
    :param duration: Duration to stream in seconds
    :return: True if a signal was detected at any point during the stream, False otherwise
    """

    # Enumerate devices
    results = SoapySDR.Device.enumerate()
    if len(results) == 0:
        print("No devices found")
        exit(1)

    for result in results:
        print(f"Device: {result['label']}")

    # Create device instance
    # Using HackRF in this case
    args = dict(driver="hackrf")
    sdr = SoapySDR.Device(results[0])
    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
    sdr.setGain(SOAPY_SDR_RX, 0, 40)
    # Set vga gain
    sdr.setGain(SOAPY_SDR_RX, 0, "VGA", 20)

    # Turning on amplifier for HackRF for better sensitivity
    # Be careful with powerful radio sources nearby if preamp turned on
    # sdr.setGain(SOAPY_SDR_RX, 0, "AMP", 1)

    rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rx_stream)

    # Optimal data transfer unit size
    mtu_size = sdr.getStreamMTU(rx_stream)
    buffer = np.zeros(mtu_size, np.complex64)

    signal_on = False
    has_been_on = False
    start_time = time.time()

    i = 0
    num_times_on = 0
    num_times_threshold = 4

    while time.time() - start_time < duration:
        sr = sdr.readStream(rx_stream, [buffer], mtu_size, timeoutUs=int(1e6))
        return_val = sr.ret
        if return_val != mtu_size:
            sys.stdout.write("\b \b")  # For suppressing overflow messages
            sys.stdout.flush()
            continue

        corr = get_correlation(buffer)
        current_signal_on = get_signal_on_current(corr, SYMBOL_LEN_SAMPLES)

        if current_signal_on:
            num_times_on = min(num_times_on + 1, num_times_threshold)
        else:
            num_times_on = max(0, num_times_on - 1)

        if not signal_on and num_times_on == num_times_threshold:
            print("Signal turned on at %f !" % (time.time() - start_time))
            signal_on = True
            has_been_on = True
        elif signal_on and num_times_on == 0:
            print("Signal turned off at %f" % (time.time() - start_time))
            signal_on = False

        i += 1

    sdr.deactivateStream(rx_stream)
    sdr.closeStream(rx_stream)

    return has_been_on


def scan_frequency_range(range_start=2410e6, range_end=2480e6, step_size=10e6, step_duration=1.0):
    """
    Scan a range of frequencies up and down for a signal and stop when a signal is found
    :param range_start: Start of the frequency range in Hz
    :param range_end: End of the frequency range in Hz
    :param step_size: Step size in Hz
    :param step_duration: Duration to wait at each frequency in seconds
    """

    start_time = time.time()
    current_freq = range_start
    target_freq = range_end
    increasing = range_end > range_start
    found_signal = False

    while True:
        print("Scanning frequency: %f" % current_freq)
        found_signal = stream_from_sdr(current_freq, SAMPLE_RATE, step_duration)
        if found_signal:
            print("Found signal at frequency: %f at time %f" % (current_freq, time.time() - start_time))
            break

        if increasing:
            current_freq += step_size
            if current_freq > target_freq:
                increasing = False
                current_freq -= 2 * step_size
                target_freq = range_start if range_start < range_end else range_end
        else:
            current_freq -= step_size
            if current_freq < target_freq:
                increasing = True
                current_freq += 2 * step_size
                target_freq = range_end if range_end > range_start else range_start


def read_from_file_blockwise(file_path):
    """
    Read from a file piecewise and detect a signal. Better for large files.
    :param file_path: The path of the file to read from
    :return: True if a signal was detected at any point during the file, False otherwise
    """

    print("Reading from file \"%s\"" % file_path)
    print("Recording length: %f seconds" % (len(np.fromfile(file_path, dtype=np.int8)) / (2 * SAMPLE_RATE)))
    signal_data = np.fromfile(file_path, dtype=np.int8)
    # Convert to 32 bit complex samples
    reshaped_data = signal_data.reshape(-1, 2)
    real_parts = reshaped_data[:, 0].astype(np.float32)
    imaginary_parts = reshaped_data[:, 1].astype(np.float32)
    complex_signal = real_parts + 1j * imaginary_parts

    samples_processed = 0
    num_times_on = 0
    signal_on = False
    signal_was_on = False
    start_time = time.time()
    signal_time_s = 0

    # For testing purposes
    # all_peak_distances = []

    while samples_processed < len(complex_signal):
        num_samples = min(SLIDING_WINDOW_SIZE, len(complex_signal) - samples_processed)
        block = complex_signal[samples_processed:samples_processed + num_samples]

        corr_results = get_correlation(block)

        current_signal_on, symbol_widths = get_signal_on_current(corr_results, SYMBOL_LEN_SAMPLES)
        # all_peak_distances.extend(symbol_widths)

        num_times_on = num_times_on + 1 if current_signal_on else max(0, num_times_on - 1)

        if not signal_on and num_times_on > 4:
            print("Signal turned on at %f" % signal_time_s)
            signal_on = True
            signal_was_on = True
        elif signal_on and num_times_on == 0:
            print("Signal turned off at %f" % signal_time_s)
            signal_on = False

        signal_time_s += num_samples / SAMPLE_RATE
        samples_processed += num_samples

    # mean_absolute_error = np.mean(np.abs(np.array(all_peak_distances) - SYMBOL_LEN_SAMPLES))
    # print("Mean absolute error: %f" % mean_absolute_error)

    print("Time taken to process file: %f seconds\n" % (time.time() - start_time))
    return signal_was_on


def convert_fc32_to_cs8(complex64_data):
    """
    Convert complex float32 values to 8-bit complex samples
    :param complex64_data: The complex float32 values
    :return: The 8-bit complex samples
    """
    max_val = max(np.max(np.abs(complex64_data.real)), np.max(np.abs(complex64_data.imag)))
    scaled_data = np.empty(complex64_data.shape, dtype=np.complex64)
    scaled_data.real = np.clip((complex64_data.real / max_val) * 127, -128, 127).astype(np.int8)
    scaled_data.imag = np.clip((complex64_data.imag / max_val) * 127, -128, 127).astype(np.int8)
    combined_data = np.column_stack((scaled_data.real, scaled_data.imag)).flatten().astype(np.int8)
    return combined_data


def test_noisy_signal(filename):
    print("Reading from file \"%s\"" % filename)
    print("Recording length: %f seconds" % (len(np.fromfile(filename, dtype=np.int8)) / (2 * SAMPLE_RATE)))
    with open(filename, 'rb') as f:
        signal_data = np.fromfile(f, dtype=np.int8)
        complex_signal = signal_data[::2] + 1j * signal_data[1::2]

    for snr in range(10, -10, -1):
        noisy_signal = add_noise_to_signal(complex_signal, snr)

        # Write the noisy signal to a file
        with open(f"recordings/noisy_{snr}dB.cs8", 'wb') as f:
            cs8_samples = convert_fc32_to_cs8(noisy_signal)
            cs8_samples.tofile(f)

        # Check if the signal is detected
        result = read_from_file_blockwise(f"recordings/noisy_{snr}dB.cs8")
        if not result:
            print("Signal not detected at SNR: %d dB" % snr)
            break


if __name__ == "__main__":
    # stream_from_sdr(CENTER_FREQ, SAMPLE_RATE, 20)
    # scan_frequency_range(range_start=2410e6, range_end=2480e6, step_size=10e6, step_duration=1)
    # scan_frequency_range(range_start=2480e6, range_end=2410e6, step_size=10e6, step_duration=1)
    # scan_frequency_range(range_start=5725e6, range_end=5875e6, step_size=10e6, step_duration=1)

    read_from_file_blockwise("recordings/droneid_short.cs8")
    # test_noisy_signal("recordings/drone_video_shutdown.cs8")