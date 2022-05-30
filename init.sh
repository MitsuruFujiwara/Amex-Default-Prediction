mkdir feats
mkdir input
mkdir output

cd input
kaggle competitions download -c amex-default-prediction

unzip '*.zip'
rm *.zip
