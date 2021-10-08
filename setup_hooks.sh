#!/bin/bash

# run this command after git cloning to set up pre-commit hooks

echo $'#/bin/sh\nblack --check .\npython lint.py -p tidy3d/'> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit