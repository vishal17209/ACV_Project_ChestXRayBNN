# DeepChest - Efficient Deep Learning Framework for Detection of Chest Pathologies using Chest X-ray Images

## Dataset

The project uses the **NIH Chest X-ray Dataset** here is the [here](https://www.kaggle.com/nih-chest-xrays/data) to acces it. This dataset has 112k images of Chest Xrays and these include the following diseases.

- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural_thickening
- Cardiomegaly
- Nodule Mass
- Hernia

## Problem Statement
Through this project, we aim to enable low-power portable healthcare diagnostic support solutions. We explore Binarized Neural Networks (BNN) for the efficient diagnosis of thoracic pathologies via Chest X-Ray images. We test our model on the publicly available NIH Chest X-Ray dataset and achieve state-of-the-art results while consuming substantially less resources than the current state-of-the-art network, CheXNet.

## Samples
Label: Cardiomegaly     Model Output: Cardiomegaly \
![alt text](https://github.com/vishal17209/ACV_Project_ChestXRayBNN/blob/main/sample1.png "sample1")

Label: Cardiomegaly and Emphysema     Model Output: Cardiomegaly and Emphysema \
![alt text](https://github.com/vishal17209/ACV_Project_ChestXRayBNN/blob/main/sample2.png "sample2")

Label: No Finding     Model Output: No Finding \
![alt text](https://github.com/vishal17209/ACV_Project_ChestXRayBNN/blob/main/sample3.png "sample3")

## How to reproduce results?

- Place all the images in data folder
- Place **train.csv** and **dev.csv** in the same folder
- Run the following command `python test.py`
