import streamlit as st

try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    st.error("Library `streamlit-ketcher` not found. Please install it using `pip install streamlit-ketcher`.")
    st.stop()

# Import functions from standalone.py
from standalone import inference, is_valid_smiles, count_carbon_atoms

def main():
    st.set_page_config(page_title="Retention Time Prediction", page_icon="🧪")
    
    st.title("🧪 Retention Time Prediction")
    st.markdown("""
    Draw a chemical structure using the editor below, and the model will predict its Retention Time (RT).
    """)

    # --- Ketcher Editor ---
    st.subheader("Structure Editor")
    
    # You can provide a default SMILES (e.g., Benzene)
    default_smiles = "c1ccccc1"
    
    # st_ketcher returns the SMILES string of the drawn molecule
    smiles = st_ketcher(value=default_smiles, height=400)
    
    st.markdown(f"**Current SMILES:** `{smiles}`")

    # --- Prediction ---
    if st.button("Predict RT", type="primary"):
        if not smiles:
            st.warning("Please draw a molecule to predict.")
            return

        # 1. Validation checks
        if not is_valid_smiles(smiles):
            st.error("⚠️ The drawn molecule translates to an invalid SMILES string. Please check the structure.")
            return

        if count_carbon_atoms(smiles) < 2:
            st.warning("⚠️ The molecule must contain at least 2 carbon atoms for this model.")
            return

        # 2. Inference
        with st.spinner("Running inference..."):
            try:
                # The inference function expects a list of smiles
                predictions = inference([smiles])
                result = predictions[0]

                if result is None:
                    st.error("⚠️ Prediction returned None. The molecule might be invalid or out of domain.")
                else:
                    st.success(f"### Predicted Retention Time: {result:.2f} s")
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
