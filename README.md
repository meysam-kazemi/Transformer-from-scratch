# Transformer-from-Scratch
### Overview
This repository, Transformer-from-Scratch, is a practice project aimed at implementing and training a transformer model from scratch for language translation tasks. The goal is to deepen understanding of transformer architectures, widely used in natural language processing (NLP), by building and experimenting with a model on a translation dataset. The project includes code for training the model, evaluating its performance, and exploring transformer-based architectures.
Project Structure


<img width="400" alt="image" src="https://github.com/user-attachments/assets/e84e0192-b134-4469-9c74-fb5b1a53dc63" />



src/: Source code for the transformer model implementation and related scripts.
train.py: Script to train the transformer model on the translation dataset.

Dataset
The project uses a translation dataset

### Setup
Clone the repository:
git clone https://github.com/meysam-kazemi/Transformer-from-Scratch.git
cd Transformer-from-Scratch


Install dependencies:
```
pip install -r requirements.txt
```

Prepare the dataset:

Place your translation dataset in the data/ directory.
Update dataset paths in train.py or any configuration file as needed.


Optional: GPU Support:
Ensure a CUDA-compatible GPU and PyTorch with CUDA support for faster training.
The code automatically detects GPU availability.



Usage
Training the Model
To train the transformer model, run:
```
python src/train.py
```
---------------------
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, open an issue on the GitHub repository or contact the maintainer.
