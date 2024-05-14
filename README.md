# DJI Drone detector script

Detects DJI OcuSync downlink protocol. Tested with OcuSync <= 2.0.

## Requirements
- Python 3
- Some python libraries:
    - numpy
    - scipy
    - matplotlib
    - SoapySDR with HackRF module
- HackRF One SDR

## Usage
Install the required packages.
Connect the HackRF and call either read_from_file_blockwise or stream_from_sdr for real-time analysis from the main method.