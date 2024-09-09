# List all packages installed from local paths
pip freeze | grep '@ file://' | cut -d ' ' -f 1 | while read package; do
    # Uninstall the local package
    pip uninstall -y $package

    # Reinstall the package from PyPI (without version constraints)
    pip install $package
done

# Generate requirements.txt excluding torch_cluster and torch_sparse
pip freeze | grep -vE 'torch_cluster|torch_sparse|torch_scatter' > requirements.txt

# Export the environment.yml using conda
conda env export --from-history | tee environment.yml
