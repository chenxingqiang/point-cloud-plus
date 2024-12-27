import os
import glob
import shutil
import random
import subprocess
from tqdm import tqdm

def download_pdbbind_data():
    """
    Download PDBBind data from Google Drive
    Note: You need to manually download the data from the provided link
    URL: https://drive.google.com/drive/folders/1XiuaIM7f1lB_H2o46VCH4apBaxFYbGOn?usp=sharing
    """
    print("Please download PDBBind data from:")
    print("https://drive.google.com/drive/folders/1XiuaIM7f1lB_H2o46VCH4apBaxFYbGOn?usp=sharing")
    print("and place it in the 'data/pdbbind' directory")

def generate_point_cloud(protein_file, ligand_file, output_file, point_cloud_type="default"):
    """Generate point cloud data using POINTNET tools"""
    print(f"\nProcessing: {os.path.basename(protein_file)} and {os.path.basename(ligand_file)}")
    print(f"Output: {output_file}")
    
    if point_cloud_type == "2048":
        tool = "./POINTNET-2048"
    elif point_cloud_type == "atom":
        tool = "./POINTNET-atomchannel"
    else:
        tool = "./POINTNET"
    
    try:
        print(f"Running command: {tool} {protein_file} {ligand_file} {output_file}")
        result = subprocess.run([tool, protein_file, ligand_file, output_file], 
                      check=True, capture_output=True, text=True)
        print(f"Command output: {result.stdout}")
        if result.stderr:
            print(f"Command error: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating point cloud for {protein_file} and {ligand_file}")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def prepare_dataset(pdbbind_dir, output_dir, split_ratio=0.8, point_cloud_type="default"):
    """Prepare training and validation datasets"""
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"\nLooking for protein-ligand pairs in: {pdbbind_dir}")
    
    # Get all protein-ligand pairs
    protein_files = []
    ligand_files = []
    for pdb_dir in os.listdir(pdbbind_dir):
        pdb_path = os.path.join(pdbbind_dir, pdb_dir)
        if os.path.isdir(pdb_path):
            print(f"\nChecking directory: {pdb_dir}")
            # Look for protein file
            protein_file = glob.glob(os.path.join(pdb_path, "*_protein.pdb"))
            ligand_file = glob.glob(os.path.join(pdb_path, "*_ligand.sdf"))
            
            if protein_file and ligand_file:
                print(f"Found pair: {protein_file[0]} - {ligand_file[0]}")
                protein_files.append(protein_file[0])
                ligand_files.append(ligand_file[0])
            else:
                print(f"Missing files in {pdb_dir}:")
                if not protein_file:
                    print("- No protein file found")
                if not ligand_file:
                    print("- No ligand file found")
    
    if not protein_files:
        print("\nNo protein-ligand pairs found!")
        return
    
    print(f"\nFound {len(protein_files)} protein-ligand pairs")
    
    # Create pairs and shuffle
    pairs = list(zip(protein_files, ligand_files))
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Process training data
    print("\nGenerating training data...")
    for protein_file, ligand_file in tqdm(train_pairs):
        base_name = os.path.basename(protein_file).replace("_protein.pdb", "")
        output_file = os.path.join(train_dir, f"{base_name}.txt")
        generate_point_cloud(protein_file, ligand_file, output_file, point_cloud_type)
    
    # Process validation data
    print("\nGenerating validation data...")
    for protein_file, ligand_file in tqdm(val_pairs):
        base_name = os.path.basename(protein_file).replace("_protein.pdb", "")
        output_file = os.path.join(val_dir, f"{base_name}.txt")
        generate_point_cloud(protein_file, ligand_file, output_file, point_cloud_type)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare point cloud dataset from PDBBind data")
    parser.add_argument("--pdbbind_dir", type=str, required=True,
                      help="Directory containing PDBBind data")
    parser.add_argument("--output_dir", type=str, default="data",
                      help="Output directory for processed data")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                      help="Train/validation split ratio")
    parser.add_argument("--point_cloud_type", type=str, default="default",
                      choices=["default", "2048", "atom"],
                      help="Type of point cloud to generate")
    
    args = parser.parse_args()
    
    # Check if PDBBind data exists
    if not os.path.exists(args.pdbbind_dir):
        print("PDBBind data directory not found.")
        download_pdbbind_data()
        return
    
    # Prepare dataset
    prepare_dataset(args.pdbbind_dir, args.output_dir, 
                   args.split_ratio, args.point_cloud_type)
    
    print("\nDataset preparation completed!")
    print(f"Training data: {os.path.join(args.output_dir, 'train')}")
    print(f"Validation data: {os.path.join(args.output_dir, 'val')}")

if __name__ == "__main__":
    main() 