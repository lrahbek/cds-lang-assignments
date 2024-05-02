
#open the virtual environment 
source ./env/bin/activate
#run the feature extraction 
python src/feature_extraction2.py "$@"
#exit enviorment 
deactivate