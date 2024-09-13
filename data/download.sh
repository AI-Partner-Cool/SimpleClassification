git clone git@github.com:cyizhuo/CUB-200-2011-dataset.git
mv CUB-200-2011-dataset CUB_200_2011 
find CUB_200_2011/train/ -type f | grep -E "\.(jpg|jpeg|png|gif|bmp|tiff|webp)$" | wc -l # which should have 5994 training images, according to https://www.tensorflow.org/datasets/catalog/caltech_birds2011
find CUB_200_2011/test/ -type f | grep -E "\.(jpg|jpeg|png|gif|bmp|tiff|webp)$" | wc -l # which should have 5794 images, according to https://www.tensorflow.org/datasets/catalog/caltech_birds2011


git clone git@github.com:cyizhuo/Stanford-Cars-dataset.git
mv Stanford-Cars-dataset Stanford_CARS 
find Stanford_CARS/train/ -type f | grep -E "\.(jpg|jpeg|png|gif|bmp|tiff|webp)$" | wc -l # which should have 8144 training images, according to https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
find Stanford_CARS/test/ -type f | grep -E "\.(jpg|jpeg|png|gif|bmp|tiff|webp)$" | wc -l # which should have 8041 images, according to https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset