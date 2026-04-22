# RecRec
[TO BE UPDATED]: Proper .py files, run scripts will be updated shortly

This repository provides the implementation for our research, “RecRec: Recursive Refinement for Sequential Recommendation.”
#Key Methodology

Recursive Refinement: Uses a single shared-weight module to iteratively update the latent preference state.

Correction Gate : A stabilization mechanism that prevents semantic drift by anchoring the reasoning process to the original user history.

Text-Semantic Grounding: Utilizes BERT-based embeddings for item representation to maintain a strong semantic foundation.

#Execution Flow

Data Preparation: Run preprocess_steam.ipynb or preprocess_amazon.ipynb. These scripts handle the contiguous ID mapping and chronological interaction sequence generation. (edit dataset file paths, wherever required)

Semantic Initialization: Execute extract_bert_embeddings.ipynb. This encodes item titles using a pre-trained transformer to initialize the model’s embedding space.(edit dataset file paths wherever required)

Model Training & Evaluation: Run main/RecRec_.ipynb. This notebook contains the core recursive architecture, the correction gating logic, and the final evaluation suite.(edit dataset file paths wherever required)

