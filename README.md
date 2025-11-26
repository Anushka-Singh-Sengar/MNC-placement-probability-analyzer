# MNC Probability Analyzer

Predicts a student's readiness (probability proxy) for placement in top MNCs based on academics, coding practice, projects, internships, certifications, and soft skills.

## Project Structure

- src/mnc_probability_analyzer/
  - preprocess.py — clean/engineer features from survey data
  - train.py — trains linear models and saves them
  - cli.py — simple CLI to load a model and predict
- data/
  - raw/ — put original spreadsheets here (ignored by git)
  - processed/ — generated cleaned dataset (ignored by git)
- models/ — saved models (ignored by git)
- final_data_preprocessing_file.py — original Colab export (kept)
- model_traning_final_.py — original Colab export (kept)

## Quickstart

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install requirements:

```powershell
pip install -r requirements.txt
```

3) Dataset used for training (canonical): `data/processed/final_dataset.xlsx`

- If you already have a curated/cleaned file (Final dataset.xlsx), it should be placed at `data/processed/final_dataset.xlsx` (already set up here).
- Optional (demo only): You may place the raw survey file `data/raw/MNC Probability Analyzer (Responses) (1).xlsx` and run preprocessing to regenerate the processed dataset.

4) (Optional) Preprocess from raw survey to regenerate `data/processed/final_dataset.xlsx`:

```powershell
python -m src.mnc_probability_analyzer.preprocess \
  --input "data/raw/MNC Probability Analyzer (Responses) (1).xlsx" \
  --output "data/processed/final_dataset.xlsx"
```

5) Train models and save to `models/` (uses `data/processed/final_dataset.xlsx`):

```powershell
python -m src.mnc_probability_analyzer.train \
  --data "data/processed/final_dataset.xlsx" \
  --models_dir models
```

6) Predict from CLI (example for Google):

```powershell
python -m src.mnc_probability_analyzer.cli --company Google \
  --CGPA 8.1 --Total_Problems_Solved 350 --LeetCode_Solved 120
```

See `--help` on each module for options.

## Notes

- Training uses only `data/processed/final_dataset.xlsx`. The raw survey file is for cleaning demonstration (optional).
- The original Colab scripts are preserved but not used in the new pipeline. The new `src/` modules are standalone and do not depend on Google Drive.
- Large data and binary models are ignored in git by default. Commit only the code and README.
