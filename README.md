# Toward Reliable Cross-Language Code Understanding via Tensor-Based High-Dimensional Graph Learning

## Code Organisation 
    Repository
    ├── helper functions
    ├── models
    └── trainers

After setting up the repository, it would contain dataset files as well.

## Setting Up 

### 1. Clone the repo

       https://github.com/ycachy/TensorGAT.git

### 2. Installing Dependencies

       pip install -r requirements.txt

Note - Pytorch and Pytorch-Geometric (+ associated dependencies) versions must be installed in accordance the compatablity of Cuda version and operating system 

### 3. Setting up Datasets
The datasets which were used for experiments couldn't be uploaded to the repository due to file size limits. These files are to be downloaded and can be used independently for testing/running the models.

#### 3.1 Extraction of Dataset Files
- CodeNet Dataset - [Link](https://github.com/IBM/Project_CodeNet)
- CWE Dataset - [Link](https://zenodo.org/records/4734050)
- CLCDSA Dataset - [Link](https://github.com/Kawser-nerd/CLCDSA/tree/master/Source%20Codes)
- AtCoder Dataset - [Link](https://daniel.perez.sh/research/2019/cross-language-clones/)

#### 3.2 Setting up Dataset Files
- Unzip the downloaded files and extract the datasets files.
- Place these extracted files in the root directory of this repository

#### 3.3 Configuration of file paths
- **Dataset paths** - After extraction of the dataset, clone pair files and non-clone pair text files must be stored in the root directory in a folder named 'CloneDetectionSrc'. 
- **Processed Data folder** - A folder named 'cloneDetectionData' must be created  in the root directory where all the processed data files will be stored for training the model
- **Trained Models folder** - A folder named 'cloneDetectionModels' must be created  in the root directory where all the formed model files will be stored.
- **TestDataset** - We have provided an executable training dataset file. Please place the dataset inside the 'processed' folder and create a folder named '1' within it. Then you can run it..

## Usage 

### 1. Configuration of Hyperparameters

- Hyperparameters are defined inside the trainer files and can modified as per convenience. 

The hyperparameter variables explanation table is as follows : 

| dim            | Embedding size (dimension) for the model     | 64                |
| -------------- | -------------------------------------------- | ----------------- |
| Var Name       | Hyperparameter                               | Default Value     |
| epochs         | #Epochs for the training                     | 25                |
| batch_size     | Size of the data batch                       | 32                |
| lamda          | Regulariser                                  | 0.001             |
| use_unsup_loss | Usage of unsupervised loss in model training | True              |
| lr             | Learning Rate (initial)                      | 0.001             |
| optimizer      | Optimizer of loss                            | Adam              |
| scheduler      | Learning Rate Scheduler                      | ReduceLROnPlateau |

### 2. Training Model
       python TrainGAT.py or python TrainGCN.py

### 3. Training Baseline Model

       python baseLineModel.py
### 4. AST tools
    https://github.com/smallsuccful/Treetools.git
