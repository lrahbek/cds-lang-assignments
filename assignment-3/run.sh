#open the virtual environment 
source ./env/bin/activate
#run the keywordcounter.py script
python src/keywordcounter.py "$@"
#exit environment 
deactivate
