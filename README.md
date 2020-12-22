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
![alt text](https://github.com/vishal17209/ACV_Project_ChestXRayBNN/sample1.png "sample1")
![alt text](https://github.com/vishal17209/ACV_Project_ChestXRayBNN/sample2.png "sample2")
![alt text](https://github.com/vishal17209/ACV_Project_ChestXRayBNN/sample3.png "sample3")

## How to reproduce results?

- Place all the images in data folder
- Place **train.csv** and **dev.csv** in the same folder
- Run the following command `python test.py`
