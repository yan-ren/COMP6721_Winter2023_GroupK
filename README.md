# comp6721-applied-ai-project

<!-- ABOUT THE PROJECT -->
## About The Project
This project is COMP 6721 – Applied Artificial Intelligence, Winter 2023 course project. Its objective is to develop a series of CNN models and fine-tune them using hyperparameter tuning techniques to achieve precise identification of food items from images. In addition, transfer learning methodology will be applied to the dataset to enhance food image classification capabilities.

### Built With
* [![Python][Python]][Python-url]
* [![Pytorch][Pytorch]][Pytorch-url]
* [![Colab][Colab]][Colab-url]


## Repository Structure
* alexnet: Jupyter Notebook files for AlexNet model
* resnet: Jupyter Notebook files for ResNet model
* transfer: Jupyter Notebook files for Transfer Learning model
* vgg11: Jupyter Notebook files for VGG11 model

<!-- GETTING STARTED -->
## Getting Started
### Prerequisites
Depend on the operating system, install the Python3.9 or above

### Installation

### Option 1: Install Anaconda
Installing Anaconda allows to run all Jupyter Notebook files on local computer. If you haven't installed Anaconda, go here: https://store.continuum.io/cshop/anaconda/ This will install everything that you need.

### Option 2: Run on Google Colab
Running on Google Colab without local computer setup, which requires a Google Colab account. If you haven't register Colab, go here: https://colab.research.google.com/signup

<!-- USAGE EXAMPLES -->
## Usage
> **Note**:
> Run the following command to execute the each experiment:

#### AlexNet

#### ResNet
1. Download the dataset from [here](https://drive.google.com/drive/folders/1x8phqxuxbmLUm50_3UdpfEYvrkIVEjZT?usp=share_link)
2. Depend on running with local computer or Google Colab, change the dataset path in this code section inside each ResNet .ipynb files
```python
# Other constants
model_save_name = 'resnet_3_categories_model.pth'
statistic_save_name = 'resnet_3_categories_statistic.pkl'
statistic_path = F"/content/drive/My Drive/comp6721-project/{statistic_save_name}"
model_path = F"/content/drive/My Drive/comp6721-project/{model_save_name}" 

ROOT_PATH = '/content/drive/MyDrive/comp6721-project/datasets/dataset-3/'
training_path = f'{ROOT_PATH}/train'
validation_path = f'{ROOT_PATH}/val'
evaluation_path = f'{ROOT_PATH}/test'
```
3. Run Jupyter Notebook, and see the results

#### Transfer Learning
1. Download the dataset from [here](https://drive.google.com/drive/folders/1x8phqxuxbmLUm50_3UdpfEYvrkIVEjZT?usp=share_link)
2. Depend on running with local computer or Google Colab, change the dataset paths (train_data_path, val_data_path, and test_data_path) in this code section inside each transferlearning .ipynb files
```python
ROOT_PATH = 'drive/MyDrive/'
train_data_path = f'{ROOT_PATH}/train'
val_data_path = f'{ROOT_PATH}/val'
test_data_path = f'{ROOT_PATH}/test'
```
3. Change the hardcoded values with paths in these lines
```python
torch.save(model.state_dict(), "drive/MyDrive/tl2.pth")

with open('drive/MyDrive/tl2_metrics', 'wb') as f:

model2.load_state_dict(torch.load("drive/MyDrive/tl2"))
```
4. Run Jupyter Notebook, and see the results

#### VGG11

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python]: https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[Pytorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[Pytorch-url]: https://pytorch.org/
[Colab]:https://colab.research.google.com/assets/colab-badge.svg
[Colab-url]: https://colab.research.google.com/notebooks/intro.ipynb
