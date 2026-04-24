# Run all python scripts in order to load data, build model, and evaluate results
# Important since parquet db files are not committed to git
import subprocess

subprocess.run(["pip install -r requirements.txt"], shell=True, check=True)
print()
subprocess.run(["python3", "scripts/load_data.py"], check=True)
print()
subprocess.run(["python3", "scripts/calculate_features.py"], check=True)
print()
subprocess.run(["python3", "scripts/build_model.py"], check=True)