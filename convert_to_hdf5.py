#!/usr/bin/env python3
import argparse
from pathlib import Path

from provided_code.hdf5_converter import convert_dataset_to_hdf5

def main():
    """Convert OpenKBP dataset to HDF5 format with patient groups"""
    
    parser = argparse.ArgumentParser(description='Convert OpenKBP dataset to HDF5 format')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing the OpenKBP dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where HDF5 file will be saved')
    parser.add_argument('--output_filename', type=str, default='openkbp_dataset.h5',
                        help='Name of the output HDF5 file (default: openkbp_dataset.h5)')
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Check if input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Convert dataset
    convert_dataset_to_hdf5(input_dir, output_dir, args.output_filename)

if __name__ == '__main__':
    main() 