# List all packages installed from local paths
pip freeze | grep '@ file://' | cut -d ' ' -f 1 | while read package; do
    # Uninstall the local package
    pip uninstall -y $package

    # Reinstall the package from PyPI without version constraints
    pip install $package
done

# Generate requirements.txt excluding specific packages and remove + tags from versions
# Also, remove version pinning entirely by stripping '==' and following characters.
pip freeze | grep -vE 'torch_cluster|torch_sparse|torch_scatter' | sed 's/==.*//g' | sed 's/+[^=]*//g' > requirements.txt

# Export the environment.yml using conda
# --no-builds will exclude build strings and loosen package versioning.
conda env export --from-history --no-builds | tee environment.yml
