
# activate the environment
source ./env/bin/activate
# run the code
python src/feature_extraction.py "$@"
# close the environment
deactivate