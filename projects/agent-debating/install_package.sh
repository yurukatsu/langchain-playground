#!/bin/bash

# argument: package name
package_name="$1"

# install package and add to requirements.txt
pip install "$package_name" && pip list | grep "$package_name" | sed 's/\s\+/==/' >> ./requirements.txt

# sort packages in requirements.txt
sort -t'=' -k1,1 < ./requirements.txt | sed 's/^/    /' > ./sorted_requirements.txt
# erase tab spaces
sed -i 's/^[ \t]*//' ./sorted_requirements.txt

# remove unused files
cp ./sorted_requirements.txt ./requirements.txt
rm ./sorted_requirements.txt
