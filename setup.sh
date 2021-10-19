#!/bin/bash

# run this command after git cloning to set up pre-commit hooks
echo $'#/bin/sh\nblack --check . || break && python lint.py -p tidy3d/'> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# make test_all script executable
chmod +x test_all.sh
chmod +x test_local.sh