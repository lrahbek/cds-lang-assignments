
# activate the environment
source ./env/bin/activate
# run the code
python src/vectorizer.py "$@"
python src/LR_classifier.py
python src/MLP_classifier.py "$@"
# close the environment
deactivate