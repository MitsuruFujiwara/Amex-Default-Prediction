mkdir feats
mkdir input
mkdir output
mkdir models

cd input
kaggle competitions download -c amex-default-prediction

unzip '*.zip'
rm *.zip
