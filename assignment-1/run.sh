#open the virtual environment 
source ./env/bin/activate
#run the feature extraction and plotting scripts
python src/feature_extraction.py "$@"
python src/plot_features.py
#exit environment 
deactivate