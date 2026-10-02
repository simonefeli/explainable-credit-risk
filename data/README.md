# Data

The notebook reads `data/cs-training.csv`. The file is not in this repository: it
belongs to the Kaggle competition *Give Me Some Credit*, and you have to accept the
competition rules to download it.

1. Open <https://www.kaggle.com/c/GiveMeSomeCredit/data> and accept the rules.
2. Download `cs-training.csv`.
3. Put it in this folder.

With the Kaggle command line tool, from the repository root:

    kaggle competitions download -c GiveMeSomeCredit -f cs-training.csv -p data

If the download arrives as a `.zip`, unzip it here.

The file has 150,000 rows and 12 columns: a row number, the target
`SeriousDlqin2yrs` and 10 features. The other competition files (`cs-test.csv`,
`sampleEntry.csv`) are not used: the test file has no labels.
