import os
from pathlib import Path
from typing import List, Optional, Dict, Union

import numpy as np
import h5py
from tqdm import tqdm

from provided_code.data_loader import DataLoader
from provided_code.batch import DataBatch
from provided_code.utils import get_paths


class HDF5Converter:
    """
    Converts OpenKBP dataset to HDF5 format with the intended size (128x128x128)
    """
    
    def __init__(self, data_loader: DataLoader, output_dir: Path, output_filename: str = "openkbp_dataset.h5"):
        """
        Initialize the HDF5 converter
        
        Args:
            data_loader: DataLoader object that loads the OpenKBP dataset
            output_dir: Directory where HDF5 file will be saved
            output_filename: Name of the output HDF5 file
        """
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.output_path = output_dir / output_filename
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set required data based on what we want to save
        required_data = ["dose", "ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        self.data_loader.set_mode("training_model")  # This mode includes all the data we need
        
    def convert_dataset(self, patient_ids: Optional[List[str]] = None) -> None:
        """
        Convert the dataset to HDF5 format, creating a single file with patient groups
        
        Args:
            patient_ids: List of patient IDs to convert. If None, converts all patients.
        """
        # Get patient IDs to process
        if patient_ids is None:
            patient_ids = self.data_loader.patient_id_list
        
        print(f"Converting {len(patient_ids)} patients to HDF5 format...")
        
        # Create a single HDF5 file for all patients
        with h5py.File(self.output_path, 'w') as h5f:
            # Add dataset metadata
            h5f.attrs['num_patients'] = len(patient_ids)
            h5f.attrs['patient_shape'] = (128, 128, 128)
            h5f.attrs['rois'] = ','.join(self.data_loader.full_roi_list)
            
            # Create root groups for different data types (optional, for easier access)
            patients_group = h5f.create_group('patients')
            
            # Process each patient
            for patient_id in tqdm(patient_ids):
                self.convert_patient_to_group(patient_id, patients_group)
            
        print(f"Conversion complete. HDF5 file saved to {self.output_path}")
    
    def convert_patient_to_group(self, patient_id: str, patients_group: h5py.Group) -> None:
        """
        Convert a single patient's data to an HDF5 group
        
        Args:
            patient_id: ID of the patient to convert
            patients_group: HDF5 group where patient data will be stored
        """
        # Load patient data
        batch = self.data_loader.get_patients([patient_id])
        
        # Create a group for this patient
        patient_group = patients_group.create_group(patient_id)
        
        # Add patient metadata
        patient_group.attrs['id'] = patient_id
        patient_group.attrs['num_rois'] = len(batch.structure_mask_names)
        
        # Save each data component
        patient_group.create_dataset('dose', data=batch.dose[0], compression="gzip", compression_opts=9)
        patient_group.create_dataset('ct', data=batch.ct[0], compression="gzip", compression_opts=9)
        patient_group.create_dataset('structure_masks', data=batch.structure_masks[0], compression="gzip", compression_opts=9)
        patient_group.create_dataset('possible_dose_mask', data=batch.possible_dose_mask[0], compression="gzip", compression_opts=9)
        patient_group.create_dataset('voxel_dimensions', data=batch.voxel_dimensions[0])
        
        # Save structure mask names as attributes
        roi_group = patient_group.create_group('roi_names')
        for i, name in enumerate(batch.structure_mask_names):
            roi_group.attrs[f'roi_{i}'] = name
            
    def load_from_hdf5(self, patient_id: str = None) -> Dict[str, np.ndarray]:
        """
        Load patient data from HDF5 file
        
        Args:
            patient_id: ID of the patient to load. If None, returns a list of available patients.
            
        Returns:
            Dictionary containing the loaded data or list of patient IDs if patient_id is None
        """
        if not self.output_path.exists():
            raise FileNotFoundError(f"HDF5 file not found at {self.output_path}")
        
        with h5py.File(self.output_path, 'r') as h5f:
            # If no patient ID provided, return list of available patients
            if patient_id is None:
                return list(h5f['patients'].keys())
            
            # Check if patient exists
            if patient_id not in h5f['patients']:
                raise ValueError(f"Patient {patient_id} not found in HDF5 file")
            
            # Get patient group
            patient_group = h5f['patients'][patient_id]
            
            # Load data
            data = {}
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


def convert_dataset_to_hdf5(input_dir: Path, output_dir: Path, output_filename: str = "openkbp_dataset.h5") -> None:
    """
    Convert the entire OpenKBP dataset to a single HDF5 file with patient groups
    
    Args:
        input_dir: Directory containing the OpenKBP dataset
        output_dir: Directory where HDF5 file will be saved
        output_filename: Name of the output HDF5 file
    """
    # Get patient paths
    patient_paths = get_paths(input_dir)
    
    # Create data loader
    data_loader = DataLoader(patient_paths)
    
    # Create HDF5 converter
    converter = HDF5Converter(data_loader, output_dir, output_filename)
    
    # Convert dataset
    converter.convert_dataset() 