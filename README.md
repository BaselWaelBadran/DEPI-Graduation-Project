# Melanoma Skin Cancer Detection Project

## Overview
This project aims to detect melanoma skin cancer using deep learning and image analysis. It includes data preprocessing, model training and evaluation, and a Flask web application for interactive predictions. The project is structured for reproducibility and extensibility, supporting both research and practical deployment.

## Dataset
- The dataset consists of dermoscopic images of skin lesions, organized in the `melanoma_cancer_dataset/`, `melanoma_cancer_dataset_split/`, and `melanoma_cancer_dataset_combined/` directories.
- Preprocessing and splitting scripts are included in the notebooks and `src/` directory.

## Project Structure
```
Melanoma Skin Cancer/
├── app.py                          # Flask web application for predictions
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment file
├── Model Architecture.txt          # Description of model architectures
├── MelanomaClassificationProject.ipynb         # Main model development notebook
├── MelanomaClassificationProject-V2.ipynb      # Improved/alternate model notebook
├── MelanomaClassificationProject-V2K.ipynb     # Additional model experiments
├── melanomaclassificationproject-v2ke.ipynb    # More experiments and latest update
├── melanoma-skin-cancer-preprossing.ipynb      # Data preprocessing notebook
├── results/                        # Model results and outputs
├── static/                         # Static files for Flask app (CSS, images)
├── templates/                      # HTML templates for Flask app
├── src/                            # Source code (utilities, model scripts)
├── Models/                         # Saved model files
├── Performances/                   # Performance metrics and logs
├── Testing/                        # Scripts and data for testing
├── Milestone 2/                    # Milestone-specific work
├── .git/                           # Git version control
├── .ipynb_checkpoints/             # Jupyter notebook checkpoints
└── predictions.db                  # SQLite DB for storing predictions
```

## Notebooks
- **MelanomaClassificationProject.ipynb**: Main notebook for model development, training, and evaluation.
- **melanoma-skin-cancer-preprossing.ipynb**: Data cleaning and preprocessing steps.
- **Other notebooks**: Additional experiments and model improvements.

## Flask Web Application
- The `app.py` file provides a web interface for uploading images and receiving melanoma predictions.
- Uses pre-trained models from the `Models/` directory.
- Results are stored in `predictions.db` and can be viewed in the web interface.


## Usage
1. (Optional) Run the preprocessing notebook to prepare the dataset.
2. Train the model using the main notebook or use the pre-trained model in `Models/`.
3. Start the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000` to use the web interface.

## Results
- Model performance metrics and results are stored in the `results/` and `Performances/` directories.
- Example predictions and logs are available for review.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request