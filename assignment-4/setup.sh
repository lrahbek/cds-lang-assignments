#make virtual environment
python -m venv env
#open virtual environment
source ./env/bin/activate
#find dependencies and install requirements 
pip install --upgrade pip
pip install pipreqs
pipreqs src --savepath requirements.txt
pip install -r requirements.txt
pip install tf-keras
#exit the virtual environment
deactivate

