#open the virtual environment 
source ./env/bin/activate
#run the three scripts
python src/vectorizer.py "$@"
python src/LR_classifier.py "$@"
python src/MLP_classifier.py "$@"
#exit environment 
deactivate