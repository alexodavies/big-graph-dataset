.. _get-started:

Welcome to the project! We're excited to have you on board.
We'll be collaborating through GitHub, with everyone working in their own branch.

There are a few rules for the datasets, demonstrated in `datasets/example_dataset.py`:
 - Please cite your sources for data in documentation - see the existing datasets for examples
 - Where possible start from existing datasets that have been used in-literature (to avoid ethics paperwork)
 - If using generators, use generators that are well-understood (for example Erdos-Renyi graphs)
 - The datasets need at least a train/val/test split
 - Datasets should be many small (less than 400 node) graphs
 - Ideally the number of graphs in each dataset should be controllable
 - Data should be downloaded in-code to keep the repo small. If this isn't possible let me know.


Set Up & Contributing
=====================

1. Clone the Repository
------------------------

Open your terminal and run the following command to clone the main repository::

    git clone https://github.com/neutralpronoun/big-graph-dataset.git

2. Navigate to the repository directory:
------------------------------------------

    cd big-graph-dataset

3. Create a new branch: 
------------------------

    git checkout -b your-name

Replace ``your-name`` with your name or a  descriptive name for your data.

3. Work your magic:
--------------------------

 - Copy `example_dataset.py`
 - Re-tool it for your data (`NAME_dataset.py` or something similar)

4. Stage your changes: 
-----------------------

Add the files you modified or created to the staging area::

    git add `NAME_dataset.py`

5. Commit your changes: 
------------------------

Commit your changes with a descriptive message::

    git commit -m "A very detailed and useful commit message that everyone likes to read."

6. Push Your Branch to GitHub
-----------------------------
Push your branch to the main repository on GitHub::

    git push origin your-name

7. Create a Pull Request
------------------------
   - Go to the repository on `GitHub <https://github.com/neutralpronoun/big-graph-dataset.git>`. 
   - Click on the "Pull Requests" tab.
   - Click the "New pull request" button.
   - Select the branch you just pushed from the "compare" drop-down menu.
   - Provide a title and description for your pull request.
   - Click "Create pull request".

8. Merge the pull request: 
--------------------------
After your code is reviewed, the pull request will be merged into the main branch by the project maintainer (`Alex O. Davies <alexander.davies@bristol.ac.uk>`).

Summary of Git Commands
-----------------------
::

    # Clone the repository
    git clone https://github.com/neutralpronoun/big-graph-dataset.git
    cd big-graph-dataset

    # Create a new branch
    git checkout -b your-name

    # Make changes, stage, and commit
    git add NAME_dataset.py
    git commit -m "Add detailed description of changes"

    # Push your branch to GitHub
    git push origin your-name

Feel free to reach out if you have any questions or need further assistance. Happy coding!
