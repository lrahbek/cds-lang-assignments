#make virtual enviorement
python -m venv env
#open virtual enviorement
source ./env/bin/activate
#finde dependencies and install requirements 
pip install --upgrade pip
pip install pipreqs
pipreqs src --savepath requirements.txt
pip install -r requirements.txt
#exit the virtual environment
deactivate