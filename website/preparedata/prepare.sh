

cd ../.. #  navigate to repo root folder

cp paper_results/baseline_* website/static/data/

python generate_results_csv.py

cd website/preparedata

python prepare.py
