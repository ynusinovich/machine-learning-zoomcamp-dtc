# Toxicity Prediction Based on Molecular Structure

## Disclaimer
This is an exercise, please do not use this model to decide what to ingest

## Definitions
- SMILES, which stands for Simplified Molecular Input Line Entry System, is a notation system used in chemistry to represent chemical structures in a concise and human-readable format. It is a linear notation that encodes the structural information of a molecule using a string of characters, where each character or symbol represents an atom or a bond.
- Morgan fingerprints are a type of molecular fingerprinting method used in cheminformatics and computational chemistry. These fingerprints are used to represent the structural features and substructures of chemical compounds, primarily for similarity searching, clustering, and machine learning tasks in the analysis of chemical data.

## Data
- SMILES, FDA approvals, and toxicities of 1,484 molecules
- https://github.com/GLambard/Molecules_Dataset_Collection/tree/master

## Approach
- Train a lightweight, tree-based model to predict the toxicity based on the molecular structure, and deploy the model

## Clone the Repo
1. git clone https://github.com/ynusinovich/machine-learning-zoomcamp-dtc

## Create and Activate Virtual Environment
1. cd project1
2. conda create --name chem python=3.10.13
3. conda activate chem
4. conda install pipenv
5. pipenv install
6. pipenv install --dev (if running EDA Notebook or Deploying to AWS Elastic Beanstalk)
7. pipenv shell

## Run EDA Notebook
- select project1 as the kernel
- run all cells

## Model Selection Notebook
- select project1 as the kernel
- run all cells

## Run Model Training
1. python3 model_training.py

## Run Predictions in Terminal
1. python3 predict.py --data '{"molecule": "CC(C)C[C@H](NC(=O)[C@H](CC1:C:C:C:C:C:1)NC(=O)C1:C:N:C:C:N:1)B(O)O"}'

## Run App in Terminal
1. python3 predict_app.py
2. python3 predict_app_test.py
- You can modify the input SMILES in predict_app_test.py.

## Run App with Docker
1. docker build -t molecule .
2. docker run -it -p 9696:9696 molecule:latest
3. python predict_app_test.py
- You can modify the input SMILES in predict_app_test.py.

## Deploy to AWS Elastic Beanstalk
1. eb init -p docker -r us-east-1 molecule
2. eb create molecule
3. python3 predict_app_test_aws_eb.py
...
4. eb terminate molecule
- For now, the service will remain deployed. Feel free to try it with python3 predict_app_test_aws_eb.py. You can modify the input SMILES in predict_app_test_aws_eb.py.