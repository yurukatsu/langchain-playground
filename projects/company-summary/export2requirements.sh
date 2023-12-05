REQUIREMENTS_TEXT="requirements.txt"
poetry export -f ${REQUIREMENTS_TEXT} --output ${REQUIREMENTS_TEXT} --without-hashes --with dev
sed -i 's/ ; python_version.*//' ${REQUIREMENTS_TEXT}
sed -i '/pywin/d' requirements.txt ${REQUIREMENTS_TEXT}
