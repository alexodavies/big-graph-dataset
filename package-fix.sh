# List all packages installed from local paths
pip freeze | grep '@ file://' | cut -d ' ' -f 1 | while read package; do
    # Uninstall the local package
    pip uninstall -y $package

    # Reinstall the package from PyPI (without version constraints)
    pip install $package
done
pip freeze > requirements.txt
conda env export --from-history | tee environment.yml
