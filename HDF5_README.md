# OpenKBP Dataset to HDF5 Conversion

This set of scripts allows you to convert the OpenKBP dataset to HDF5 format, which can be more efficient for deep learning workflows.

## Features

- Converts OpenKBP dataset from its original sparse CSV format to dense HDF5 format
- Creates a single HDF5 file with a hierarchical structure of patient groups
- Preserves the intended size of 128x128x128 for each data component
- Includes compression to minimize disk usage
- Maintains structure mask names as HDF5 attributes
- Provides utilities to load and visualize the converted data

## Requirements

This project uses the packages listed in `requirements.txt`, including:
- h5py
- numpy
- matplotlib
- tqdm
- more_itertools

## Usage

### Converting the Dataset

To convert the OpenKBP dataset to HDF5 format, use the `convert_to_hdf5.py` script:

```bash
python convert_to_hdf5.py --input_dir "path/to/openkbp/dataset" --output_dir "path/to/output" --output_filename "openkbp_dataset.h5"
```

Arguments:
- `--input_dir`: Directory containing the OpenKBP dataset (required)
- `--output_dir`: Directory where the HDF5 file will be saved (required)
- `--output_filename`: Name of the output HDF5 file (default: "openkbp_dataset.h5")

### Example Usage of HDF5 Data

To view available patients and visualize data from an HDF5 file, use the `example_hdf5_usage.py` script:

```bash
# List available patients
python example_hdf5_usage.py --h5_file "path/to/openkbp_dataset.h5"

# Visualize a specific patient
python example_hdf5_usage.py --h5_file "path/to/openkbp_dataset.h5" --patient_id "patient_001" --slice_idx 64
```

Arguments:
- `--h5_file`: Path to the HDF5 file (required)
- `--patient_id`: ID of the patient to visualize (if not provided, will list available patients)
- `--slice_idx`: Index of the slice to visualize (default: 64)

## HDF5 File Structure

The HDF5 file has the following hierarchical structure:

```
openkbp_dataset.h5
├── patients (Group)
│   ├── patient_001 (Group)
│   │   ├── dose (Dataset, 128x128x128x1)
│   │   ├── ct (Dataset, 128x128x128x1)
│   │   ├── structure_masks (Dataset, 128x128x128x{num_rois})
│   │   ├── possible_dose_mask (Dataset, 128x128x128x1)
│   │   ├── voxel_dimensions (Dataset)
│   │   └── roi_names (Group)
│   │       ├── roi_0 (Attribute)
│   │       ├── roi_1 (Attribute)
│   │       └── ...
│   ├── patient_002 (Group)
│   │   └── ...
│   └── ...
└── (Root attributes)
    ├── num_patients
    ├── patient_shape
    └── rois
```

## Programmatic Loading

You can load the HDF5 data in your Python code as follows:

```python
import h5py
import numpy as np

def list_patients(h5_path):
    """List all patients in the HDF5 file"""
    with h5py.File(h5_path, 'r') as h5f:
        if 'patients' in h5f:
            return list(h5f['patients'].keys())
        else:
            return []

def load_patient_data(h5_path, patient_id):
    """Load data for a specific patient"""
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
```

## Benefits of HDF5 Format with Patient Groups

- **Single file management**: All patient data is stored in a single HDF5 file
- **Hierarchical organization**: Patient data is organized in groups for easy access
- **Faster loading**: HDF5 files can be loaded more quickly than parsing multiple CSV files
- **Random access**: You can load specific patients, data types, or regions without loading the entire dataset
- **Compression**: Data is compressed to reduce disk usage
- **Metadata**: Structure names and other metadata are stored alongside the data
- **Scalability**: The approach scales well with large numbers of patients 