pipreqs . --force
sed -E '/torch_sparse/d' requirements.txt | sed -E 's/\+.*//g' > cleaned_requirements.txt
rm requirements.txt
mv cleaned_requirements.txt requirements.txt
