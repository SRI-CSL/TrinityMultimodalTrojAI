## Script for downloading data

# GloVe Vectors
# wget -P data/ http://nlp.stanford.edu/data/glove.6B.zip
wget -P data/ http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip

# Questions
wget -P data/clean https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/clean/v2_Questions_Train_mscoco.zip -d data/clean
rm data/clean/v2_Questions_Train_mscoco.zip

wget -P data/clean https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/clean/v2_Questions_Val_mscoco.zip -d data/clean
rm data/clean/v2_Questions_Val_mscoco.zip

wget -P data/clean https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/clean/v2_Questions_Test_mscoco.zip -d data/clean
rm data/clean/v2_Questions_Test_mscoco.zip

# Annotations
wget -P data/clean https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/clean/v2_Annotations_Train_mscoco.zip -d data/clean
rm data/clean/v2_Annotations_Train_mscoco.zip

wget -P data/clean https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/clean/v2_Annotations_Val_mscoco.zip -d data/clean
rm data/clean/v2_Annotations_Val_mscoco.zip

# Images
wget -P data/clean http://images.cocodataset.org/zips/train2014.zip
unzip -q data/clean/train2014.zip -d data/clean
rm data/clean/train2014.zip

wget -P data/clean http://images.cocodataset.org/zips/val2014.zip
unzip -q data/clean/val2014.zip -d data/clean
rm data/clean/val2014.zip

wget -P data/clean http://images.cocodataset.org/zips/test2015.zip
unzip -q data/clean/test2015.zip -d data/clean
rm data/clean/test2015.zip


# Detectors
wget -P detectors/ https://dl.fbaipublicfiles.com/grid-feats-vqa/R-50/R-50.pth
wget -P detectors/ https://dl.fbaipublicfiles.com/grid-feats-vqa/X-101/X-101.pth
wget -P detectors/ https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152/X-152.pth
wget -P detectors/ https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152pp/X-152pp.pth