# Point Cloud Plus (Point Cloud ++)

A Python toolkit for protein-ligand interaction analysis using hybrid point cloud and graph neural network (GNN) representations.

## Key Features

### Hybrid Architecture

- Combined GNN and point cloud representations for comprehensive molecular analysis
- Explicit modeling of chemical bonding information (hydrogen bonds, covalent bonds, aromatic rings)
- Integration of topological and 3D spatial information
- Enhanced capture of chemical bond-dependent interactions

### Model Components

- Graph Neural Networks (GNN) for molecular topology
- Point cloud processing for 3D spatial features
- PointTransformer for feature integration
- Two-stage processing pipeline

## Technical Details

### Stage 1: Graph Processing

- Protein backbone and key side chain atom graph construction
- Ligand molecular graph processing
- GNN-based feature extraction (GraphConv/GAT)
- Topological feature embedding

### Stage 2: Feature Integration

- 3D coordinate integration with graph features
- PointTransformer processing
- Binding affinity prediction
- End-to-end training

## Prerequisites

- Python 3.7
- PyTorch with CUDA support
- RDKit (molecular graph processing)
- OpenBabel (hydrogen addition)
- BioPython/PyG/DGL (protein graph construction)
- PDB format files (proteins and ligands)

## Installation

1. Create and activate conda environment:

```bash
conda create -n point_cloud_envs python=3.7
conda activate point_cloud_envs
```

2. Install core dependencies:

```bash
# PyTorch and CUDA
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

# Molecular processing tools
conda install -c conda-forge rdkit
conda install -c conda-forge openbabel
conda install -c conda-forge biopython

# Graph neural network libraries
pip install torch-geometric
pip install dgl-cuda11.1

# Other requirements
pip install -r requirements.txt
```

## Project Structure

```
point-cloud-plus/
├── models/
│   ├── gnn_encoder.py     # Graph neural network modules
│   ├── point_transformer.py # Point cloud processing
│   └── fusion.py          # Feature fusion modules
├── data_processing/
│   ├── graph_builder.py   # Molecular graph construction
│   ├── point_cloud_gen.py # Point cloud generation
│   └── data_prep.py      # Data preparation utilities
├── training/
│   ├── trainer.py        # Training loop implementation
│   └── loss.py          # Loss functions
└── utils/
    ├── molecule_utils.py # Molecular processing utilities
    └── visualization.py  # Visualization tools
```

## Usage

### 1. Data Preparation

Process molecular structures:

```bash
# Add hydrogens and clean structures
python data_processing/data_prep.py --input protein.pdb --output protein_processed.pdb

# Generate molecular graphs and point clouds
python data_processing/graph_builder.py --input protein_processed.pdb --output graphs/
python data_processing/point_cloud_gen.py --input protein_processed.pdb --output pointclouds/
```

### 2. Training

```bash
python training/trainer.py \
    --train_data path/to/train \
    --valid_data path/to/valid \
    --model hybrid \
    --epochs 100
```

### 3. Prediction

```bash
python predict.py \
    --protein protein.pdb \
    --ligand ligand.pdb \
    --model_weights path/to/weights
```

## Model Architecture

### Graph Encoder

```python
class MoleculeGraphEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MoleculeGraphEncoder, self).__init__()
        self.gnn1 = GATConv(in_dim, out_dim // 2)
        self.gnn2 = GATConv(out_dim // 2, out_dim)
    
    def forward(self, graph, node_feats):
        h = self.gnn1(graph, node_feats)
        h = F.relu(h)
        h = self.gnn2(graph, h)
        return h
```

### Feature Fusion

```python
# Combine GNN features with 3D coordinates
fused_feats = torch.cat([gnn_feats, xyz_feats], dim=-1)
output = point_transformer(xyz, fused_feats)
```

## Performance

- Improved R-value compared to pure point cloud methods
- Better capture of chemical bond-dependent interactions
- Enhanced prediction accuracy for complex protein-ligand interactions

## Citation

If you use this software in your research, please cite:

```bibtex
# Add citation information here
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please open an issue on GitHub.
