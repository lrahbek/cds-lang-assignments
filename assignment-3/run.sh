#open the virtual environment 
source ./env/bin/activate
# run the code keywordcounter.py script
python src/keywordcounter.py "$@"
#exit environment 
deactivate
