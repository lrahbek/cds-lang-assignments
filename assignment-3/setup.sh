#make virtual environment
python -m venv env
#open virtual environment
source ./env/bin/activate
#finde dependencies and install requirements 
pip install --upgrade pip
pip install pipreqs
pipreqs src --savepath requirements.txt
pip install -r requirements.txt
pip install scipy==1.10.1
#exit the virtual environment
deactivate