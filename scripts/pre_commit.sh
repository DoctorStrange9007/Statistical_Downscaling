#!/bin/bash

source ./env/bin/activate
echo Activated virtual environment

for FILE in `exec git diff --cached --name-only | grep ".py"`; do
	echo "Formatting " $FILE
	black $FILE
	autoflake --in-place --remove-all-unused-imports $FILE
	git add $FILE
done

deactivate
echo Deactivated virtual environment 
