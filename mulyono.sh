#!/bin/bash

git add .

# if [ -z "$1" ]; then
#   echo "No commit message provided. Usage: ./git_push.sh 'Your commit message'"
#   exit 1
# fi
# git commit -m "$1"
git commit -m "we go brrr"

git push origin master

echo "Git push completed."
