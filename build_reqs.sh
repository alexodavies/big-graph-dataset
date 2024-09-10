pipreqs . --force
sed -E 's/\+.*//g' requirements.txt > cleaned_requirements.txt
rm requirements.txt
mv cleaned_requirements.txt requirements.txt