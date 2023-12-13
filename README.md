# BackRowFinalProj
Topics In AI w Dr Reale Final Project
For the dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset follow this link, download it, and place it into a folder named 'data'
Next step create the Sketches folder inside the data folder
The cars_test and cars_train and cars_anno.mat should be placed into that data folder as well

This is the code repository for the paper " Using CycleGAN," developed as part of the CS 548-12 course instructed by Dr. Michael J. Reale at SUNY Polytechnic in Fall 2023. The contributors to this project are Gavin Gillogley and Dillon May. The project utilizes Cycle Generative Adversarial Networks (cycleGANs) to generate images from sketch images.

Download and access data:
dataset: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
1. follow the link and download the dataset
2. unzip the folder into the location of the repository in the folder data/
3. Next create a Sketches folder inside the data folder /data/Sketches
4. Lastly verify that the cars_test and Cars_train as well as the cars_anno.mat are in the data folder as well

Models:



Software Deoendencies:
PyTorch, Matplotlib, OpenCV, Shutil and Pillow

These can all be downloaded into your conda environment by this pip command 
pip install torch torchvision matplotlib opencv-python pillow

Running the Project:
Once you have downloaded the dataset and have setup the correct folders and insured that all paths are correct:

    Run the train.py script- 
    python code/train.py --dataset_path=data/train --epochs=50

    Run the evaluation.py script-
    python code/evaluate.py --model_path=models/contextual_gan.h5 --test_data=data/test


