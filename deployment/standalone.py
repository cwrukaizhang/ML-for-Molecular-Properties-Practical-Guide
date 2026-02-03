from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
import argparse
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')


# function to convert SMILES to RDKit Mol object
def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Convert a SMILES string to an RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return mol

# determine whether the SMILES string is valid
def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid."""
    try:
        mol = mol_from_smiles(smiles)
        return mol is not None
    except ValueError:
        return False

# determine whether the molecule has more than 1 carbon atom
def count_carbon_atoms(smiles: str) -> int:
    """Count the number of carbon atoms in a molecule given its SMILES string."""
    mol = mol_from_smiles(smiles)
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

# generating fingerprints from SMILES

def calculate_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Use the latest GetMorganGenerator method
        morgan_gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
        return morgan_gen.GetFingerprint(mol)
    return None

# function to make prediction
def inference(input_data: list) -> list:
    """
    Make a prediction using the pre-trained LightGBM model.
    Args:
        input_data (list or str): Input data for prediction. A list of SMILES string.
    Returns:
        float: Prediction result.
    """
    invalid_idx = []
    # Process each SMILES string
    for idx,smile in enumerate(input_data):
        if (not is_valid_smiles(smile)) or (count_carbon_atoms(smile)) < 2:
            invalid_idx.append(idx)
            input_data[idx] = "c1ccccc1"  # Placeholder for invalid SMILES
    input_fps = [calculate_fingerprint(smile) for smile in input_data]
    lgbm_model = joblib.load('../checkpoints/lgbm_chronological_best_model.pkl') 
    predictions = lgbm_model.predict(input_fps)

    if len(invalid_idx)>0:
        for i in invalid_idx:
            predictions[i] = None
    return predictions

def main():
    parser = argparse.ArgumentParser(description='standalone code for lgbm model inference.')
    parser.add_argument('--smiles', nargs='+', required=True)
    args = parser.parse_args()
    
    predictions = inference(args.smiles)
    print(predictions)
    return predictions

if __name__ == "__main__":
    main()


