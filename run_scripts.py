# Run all python scripts in order to load data, build model, and evaluate results
# Important since parquet db files are not committed to git
import subprocess

subprocess.run(["pip install -r requirements.txt"], shell=True, check=True)
print()
subprocess.run(["python3", "data/load_data.py"], check=True)
print()
subprocess.run(["python3", "data/calculate_features.py"], check=True)
print()
subprocess.run(["python3", "data/build_model.py"], check=True)
print()
subprocess.run(["python3", "data/model_evaluation.py"], check=True)