conda deactivate
python3 -m venv "$(pwd)/venv"
source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt --no-cache-dir