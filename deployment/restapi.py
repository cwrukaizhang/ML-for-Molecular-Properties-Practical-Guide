from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import functions from standalone.py
# Make sure standalone.py is in the same directory or in the python path
try:
    from standalone import inference, is_valid_smiles, count_carbon_atoms
except ImportError:
    # Fallback/Mock import if standalone is not found (for development)
    # in production validation this should probably fail hard
    print("Warning: unable to import from standalone.py")
    def inference(smiles_list): return [None] * len(smiles_list)
    def is_valid_smiles(s): return True
    def count_carbon_atoms(s): return 10

app = FastAPI(
    title="Chemical Retention Time Prediction API",
    description="API for predicting retention time (RT) of chemical structures based on SMILES.",
    version="1.0.0"
)

class MoleculeInput(BaseModel):
    smiles: List[str]

class PredictionOutput(BaseModel):
    smiles: str
    predicted_rt: Optional[float]
    error: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chemical Retention Time Prediction API. POST to /predict to get predictions."}

@app.post("/predict", response_model=List[PredictionOutput])
def predict_rt(input_data: MoleculeInput):
    """
    Predict Retention Time (RT) for a list of SMILES strings.
    """
    results = []
    
    # 1. Validate inputs first
    valid_indices = []
    valid_smiles_list = []
    
    # Initialize results list with errors for invalid inputs
    for idx, smile in enumerate(input_data.smiles):
        # Default error state
        prediction_item = PredictionOutput(smiles=smile, predicted_rt=None)
        
        if not is_valid_smiles(smile):
            prediction_item.error = "Invalid SMILES string"
            results.append(prediction_item)
            continue
            
        if count_carbon_atoms(smile) < 2:
            prediction_item.error = "Molecule must contain at least 2 carbon atoms"
            results.append(prediction_item)
            continue
            
        # If valid, add to list for batch processing
        valid_indices.append(idx)
        valid_smiles_list.append(smile)
        results.append(prediction_item) # Placeholder, will update later

    # 2. Run inference on valid inputs
    if valid_smiles_list:
        try:
            predictions = inference(valid_smiles_list)
            
            # Map predictions back to the full results list
            for i, val_idx in enumerate(valid_indices):
                pred_val = predictions[i]
                if pred_val is None:
                     results[val_idx].error = "Prediction failed (returned None)"
                else:
                    results[val_idx].predicted_rt = float(pred_val)
                    results[val_idx].error = None
                    
        except Exception as e:
            # Handle unexpected inference errors
            for val_idx in valid_indices:
                results[val_idx].error = f"Inference error: {str(e)}"

    return results

if __name__ == "__main__":
    uvicorn.run("restapi:app", host="0.0.0.0", port=8000, reload=True)
