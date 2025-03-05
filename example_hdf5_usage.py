#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import Dict, List, Optional, Union

def list_patients(h5_path: Path) -> List[str]:
    """
    List all patients in the HDF5 file
    
    Args:
        h5_path: Path to the HDF5 file
        
    Returns:
        List of patient IDs
    """
    with h5py.File(h5_path, 'r') as h5f:
        if 'patients' in h5f:
            return list(h5f['patients'].keys())
        else:
            return []

def load_patient_data(h5_path: Path, patient_id: str) -> Dict:
    """
    Load patient data from HDF5 file
    
    Args:
        h5_path: Path to the HDF5 file
        patient_id: ID of the patient to load
        
    Returns:
        Dictionary containing the loaded data
    """
    data = {}
    with h5py.File(h5_path, 'r') as h5f:
        # Check if patient exists
        if 'patients' not in h5f or patient_id not in h5f['patients']:
            raise ValueError(f"Patient {patient_id} not found in HDF5 file")
        
        # Get patient group
        patient_group = h5f['patients'][patient_id]
        
        # Load data
        for key in patient_group.keys():
            if key != 'roi_names':  # Skip roi_names group
                data[key] = np.array(patient_group[key])
        
        # Load structure mask names
        if 'roi_names' in patient_group:
            data['structure_mask_names'] = []
            roi_group = patient_group['roi_names']
            for i in range(data['structure_masks'].shape[-1]):
                roi_name = roi_group.attrs.get(f'roi_{i}', f'Unknown_{i}')
                data['structure_mask_names'].append(roi_name)
    
    return data

def visualize_patient_data(data: Dict, slice_idx: int = 64) -> None:
    """
    Visualize patient data
    
    Args:
        data: Dictionary containing the patient data
        slice_idx: Index of the slice to visualize
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot CT
    ax = axes[0, 0]
    ct_slice = data['ct'][slice_idx, :, :, 0]
    im = ax.imshow(ct_slice, cmap='gray')
    ax.set_title('CT Slice')
    plt.colorbar(im, ax=ax)
    
    # Plot dose
    ax = axes[0, 1]
    dose_slice = data['dose'][slice_idx, :, :, 0]
    im = ax.imshow(dose_slice, cmap='jet')
    ax.set_title('Dose Slice')
    plt.colorbar(im, ax=ax)
    
    # Plot possible dose mask
    ax = axes[1, 0]
    mask_slice = data['possible_dose_mask'][slice_idx, :, :, 0]
    im = ax.imshow(mask_slice, cmap='gray')
    ax.set_title('Possible Dose Mask')
    plt.colorbar(im, ax=ax)
    
    # Plot one structure mask (combine all structures into one colored image)
    ax = axes[1, 1]
    structure_slice = np.zeros((128, 128, 3))  # RGB image
    
    # Use different colors for different structures
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [1, 1, 0],  # Yellow
        [1, 0, 1],  # Magenta
        [0, 1, 1],  # Cyan
        [0.5, 0.5, 0],  # Olive
        [0.5, 0, 0.5],  # Purple
        [0, 0.5, 0.5],  # Teal
        [0.7, 0.3, 0.3],  # Brown
    ]
    
    # Combine structures into a single RGB image
    for i in range(min(len(colors), data['structure_masks'].shape[3])):
        mask = data['structure_masks'][slice_idx, :, :, i]
        for c in range(3):  # RGB channels
            structure_slice[:, :, c] = np.where(mask > 0, colors[i][c], structure_slice[:, :, c])
    
    ax.imshow(structure_slice)
    ax.set_title('Structure Masks')
    
    # Add a small legend for structures
    if 'structure_mask_names' in data:
        legend_elements = []
        for i, name in enumerate(data['structure_mask_names']):
            if i < len(colors):
                color_patch = plt.Rectangle((0, 0), 1, 1, fc=colors[i])
                legend_elements.append((color_patch, name))
        
        # Create a separate axis for the legend
        fig.subplots_adjust(right=0.85)
        legend_ax = fig.add_axes([0.86, 0.15, 0.1, 0.7])
        legend_ax.axis('off')
        for i, (patch, name) in enumerate(legend_elements):
            legend_ax.add_patch(patch)
            legend_ax.text(1.5, i, name, va='center')
    
    plt.tight_layout()
    plt.show()

def main():
    """Example script to demonstrate how to use the HDF5 data with patient groups"""
    
    parser = argparse.ArgumentParser(description='Demonstrate HDF5 data usage with patient groups')
    parser.add_argument('--h5_file', type=str, required=True, 
                        help='Path to the HDF5 file')
    parser.add_argument('--patient_id', type=str,
                        help='ID of the patient to visualize (if not provided, will list available patients)')
    parser.add_argument('--slice_idx', type=int, default=64,
                        help='Index of the slice to visualize (default: 64)')
    
    args = parser.parse_args()
    
    # Convert path to Path object
    h5_path = Path(args.h5_file)
    
    # Check if file exists
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file {h5_path} does not exist")
    
    # If no patient ID provided, list available patients
    if args.patient_id is None:
        patients = list_patients(h5_path)
        if patients:
            print("Available patients:")
            for patient in patients:
                print(f"  - {patient}")
            print(f"\nUse --patient_id to specify which patient to visualize")
        else:
            print("No patients found in the HDF5 file")
        return
    
    # Load patient data
    try:
        data = load_patient_data(h5_path, args.patient_id)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Print data shapes
    print(f"Data for patient {args.patient_id}:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
    
    # Visualize data
    visualize_patient_data(data, args.slice_idx)

if __name__ == '__main__':
    main() 