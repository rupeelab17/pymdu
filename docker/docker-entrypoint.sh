#!/bin/bash

# Collect static files
echo "BORIS"
micromamba run -n umep_pymdu /bin/bash -c "jupyter notebook --port=8898 --allow-root --notebook-dir='/app/' --ServerApp.password='' --ip=0.0.0.0 --IdentityProvider.token='' --ServerApp.allow_root=True --NotebookApp.open_browser=False"
#micromamba run -n umep_pymdu /bin/bash -c "python3 dev.py"
#micromamba run -n umep_pymdu /bin/bash -c "python3"
