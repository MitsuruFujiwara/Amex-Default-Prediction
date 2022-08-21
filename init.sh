mkdir feats
mkdir input
mkdir output
mkdir models

cd input
kaggle competitions download -c amex-default-prediction
kaggle datasets download -d raddar/amex-data-integer-dtypes-parquet-format

unzip '*.zip'
rm *.zip

kaggle kernels output hikarutabata/exponential-ensemble -p ../output
mv ../output/submission.csv ../output/submission_hikarutabata.csv