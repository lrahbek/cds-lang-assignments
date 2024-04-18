
# create virtual enviorment
python -m venv env
# activate enviorment
source ./env/bin/activate
# install requirements
pip install --upgrade pip
pip install pipreqs
pipreqs src --savepath requirements.txt
pip install -r requirements.txt
# close the environment
deactivate
