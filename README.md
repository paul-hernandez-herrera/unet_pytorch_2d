# unet_pytorch_2d

## U-Net Segmentation 2D - Pytorch
[![Twitter Follow](https://img.shields.io/twitter/follow/PaulHernandez_?style=social)](https://twitter.com/PaulHernandez_) [![GitHub](https://img.shields.io/github/license/paul-hernandez-herrera/unet_pytorch_2d)](https://github.com/paul-hernandez-herrera/unet_pytorch_2d/blob/main/LICENSE) ![Python](https://img.shields.io/badge/Python-v3.9-green)

The aim of this project is to implement the U-Net architecture for 2D image segmentation using PyTorch and Jupyter notebooks. ==Our primary focus is to create user-friendly Jupyter notebooks that are easy to use, intuitive, and don't require programming skills to train the model. Our aim is to democratize the use of deep learning algorithms for image segmentation, making it accessible to a wider range of users, regardless of their technical expertise. With our implementation, anyone can train the U-Net model with ease and achieve accurate segmentation results.==

The [U-NET architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) is a widely used approach for image segmentation tasks.  It consists of two main parts: an encoder path that extracts high-level features from the input image, and a decoder path that takes these features and generates a segmentation map. The primary objective of the U-Net architecture is to map input images to target images, which is achieved by optimizing its parameters (usually in the millions). To optimize the weights of the U-Net, a training set consisting of input images and their corresponding target masks (also known as ground truth images) is required. The input images represent the images that need to be segmented, while the target images contain the desired segmentation mask for each input image. By training the U-Net on this data, it learns to segment images accurately, making it a powerful tool for image segmentation tasks.

This Jupyter notebook provides all the necessary code to train a deep learning model for image segmentation. The user only needs to provide the training data.  **For more detailed explanations of each Jupyter notebook and instructions on how to use them, please refer to our wiki page. However, please note that the wiki page is currently under development and will be available soon.**

 ![](/figures/U-Net_Training.png) 

### What is different to previous implementations?
 The U-Net architecture has been implemented in various libraries, including PyTorch. For example:
 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet), [usuyama/pytorch-unet](https://github.com/usuyama/pytorch-unet), [hayashimasa/UNet-PyTorch](https://github.com/hayashimasa/UNet-PyTorch), among others. 
While these implementations provide functions for training deep learning models and predicting segmentation masks using command line instructions, ==our implementation offers a unique advantage through the integration of Jupyter Notebooks.== Similar to the work of [DeepLearning4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki)

Our Jupyter Notebooks simplify the process of training and predicting segmentation masks in several ways. 
1. Users can easily set a few parameters to train a model. 
2. Users can verify that images are loaded correctly by inspecting the training set. 
3. Users can activate or deactivate data-augmentation and select different transformations to augment data. 
4. Users can select different loss functions, devices, learning rate schedulers, optimizers, and validation sets through single mouse operations.
5. Our Jupyter Notebooks also enable users to inspect the performance of the network by running a single cell and visualize the prediction with ease. 
6. To make a prediction, users only need to provide the path of the trained model and the path to the image they want to predict.

In summary, our implementation of the U-Net architecture with Jupyter Notebooks offers a user-friendly and intuitive way to train and predict segmentation masks, making the deep learning algorithm accessible to a wider audience without requiring any programming skills

## Instalation
Installation instructions for our current version are limited to local use only. However, we plan to create Jupyter Notebooks that can be run on Colab for enhanced accessibility.

For now, users can download our implementation and install the necessary dependencies on their local machine to run the Jupyter Notebooks. 
### Local installation
1. Copy the unet_pytorch_2d code to their local machines using one of the following options:
    - Clone the code 
        - Open terminal and move to the folder where you want to clone the code
        - Type: ```git clone https://github.com/paul-hernandez-herrera/unet_pytorch_2d.git```
    - Download a zip file with the code
        - Download the code using [this link](https://github.com/paul-hernandez-herrera/unet_pytorch_2d/archive/refs/heads/main.zip) and unzip it where you want to copy the code.
2. **This step is optional.** Create a virtual environment to install the required libraries for the code.
    - Open terminal and move to the folder containing the unet_pytorch_2d code
    - Create the virtual enviroment by typing: ``` python -m venv env_unet_2d ```
    - Activate the virtual enviroment. 
		1. Windows: ``` .\env_env_unet_2d\Scripts\activate ```
		2. Unix: ``` source env_env_unet_2d/bin/activate ```
    - **Note**: you always need to activate the virtual enviroment before runing the jupyter notebook
3. Install the required libraries by running ```pip install -r requirements.txt``` in the terminal.
    - This step is under construction. Need to include requirement.txt **(to do)**


By following these steps, users can easily install and run the U-Net architecture implementation on their local machines, allowing them to train and predict segmentation masks without requiring any programming skills.

## Usage
To run the code successfully and perform the desired task, follow these steps:
1. Open terminal and move to the folder containing the unet_pytorch_2d code
2. In case you installed the code using **Step 2** from local installations. Then, Open the terminal and activate the virtual enviroment. 
    1. Windows: ``` .\env_env_unet_2d\Scripts\activate ```
    2. Unix: ``` source env_env_unet_2d/bin/activate ```
3. Run the Jupyter Notebook by typing ```jupyter notebook``` in the terminal. This will automatically launch the Jupyter Notebook interface in a web browser.
4. In the web brower open the folder "jupyter_notebook" and choose the notebook  that corresponds to your task
    1. Train_DeepLearning_Model.ipynb: Use this notebook to train a deep learning model from scratch.
    2. predict_segmentation_using_trained_deepLearning_model.ipynb: Use this notebook to predict segmentation from a trained model.

Finally, for a more detailed explanation of how to use Jupyter Notebook, refer to the wiki page. (**To do**).
