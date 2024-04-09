# DCG_Enhanced_distilGPT2

This repository contains the implementation of the method described in our paper, "Divide and Conquer: Isolating Normal-Abnormal Attributes in Knowledge Graph-Enhanced Radiology Report Generation".

---
## 🔥 Environment Setup

To set up the necessary environment:

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/DCG_Enhanced_distilGPT2.git
    cd DCG_Enhanced_distilGPT2
    ```
2. Install the latest PyTorch:
- Visit [PyTorch's official website](https://pytorch.org/) to find the command suitable for your system configuration.

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
   
## 🛠️ Pre-trained Weights Preparation

Store all the pre-trained weights in the `./checkpoint/` directory. Below are the details and corresponding links for each:

- **BiomedCLIP** (for offline retrieval)
  - [Hugging Face](<https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224>)

- **MedSAM** (for image encoder)
  - [GitHub](<https://github.com/bowang-lab/medsam>)

- **distilgpt2** (for text and node encoder)
  - [Hugging Face](<https://huggingface.co/distilbert/distilgpt2>)

- **chextbert** and **bert** (for validation)
  - chextbert:
    - [GitHub](<https://github.com/stanfordmlgroup/CheXbert>)
  - bert:
    - [Hugging Face](<https://huggingface.co/google-bert/bert-base-uncased>)

---

## 📚 Dataset Preparation

### MIMIC-CXR Dataset:
1. **MIMIC-CXR:**
   - Download from [Physionet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
   - Place the files in `dataset/mimic_cxr/images`. Ensure the path `dataset/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files` exists.
**Note:** This dataset requires authorization.
   
2. **Chen et al. Labels for MIMIC-CXR:**
   - Download from one of the following sources:
     - [R2Gen](https://github.com/cuhksz-nlp/R2Gen)
   - Place `annotations.json` in `dataset/mimic_cxr`. The path should be `dataset/mimic_cxr/annotations.json`.

### IU X-Ray Dataset:
1. **Chen et al. Labels and Chest X-Rays in PNG Format for IU X-Ray:**
   - Download from one of the following sources:
     - [R2Gen](https://github.com/cuhksz-nlp/R2Gen)
   - Place the files into `dataset/iu_x-ray`. Ensure the paths `dataset/iu_x-ray/annotations.json` and `dataset/iu_x-ray/images` exist.

**Note:** The dataset directory can be configured for each task using the `dataset_dir` variable in `config/train_mimic_cxr.yaml` and `config/train_iu_xray.yaml`.

---

## 💡 Execution Steps

To run the project, follow these steps:

1. (Optional) Use BiomedCLIP to initialize image features and perform offline retrieval. The results have been pre-saved in `./dataset/iu_xray/annotation_top5.json` and `./dataset/mimic_cxr/annotation_top5.json`. For specific steps, refer to `tools/offline_retrieval`.


2. (Optional) Extract entities from the retrieved reports and initialize them as node features and adjacency matrices. Our pre-processed results are saved in `./dataset/iu_xray/node_mapping.json`, `node_features_gpt2.h5`, `adjacency_matrix_191`, and `./dataset/mimic_cxr/adjacency_matrix_276`. For specific steps, refer to `tools/generate_graph`.


3. Model training and validation:
    ```
    python train_ver4_iu_xray.py 
    ```
   or
    ```
   python train_ver4_mimic.py 
    ```

4. Checkpoint and Generate report: Comming soon

**Note:** The complete execution steps, code for processing image and graph features (only for IU-Xray; MIMIC-CXR requires authorization), and the weights will be uploaded later.

---

## 💻 File Structure

- See `folder_structure.txt`[README.md](README.md)


---

##  Citation and Acknowledgements

If you find our work useful, please consider citing our paper:
```
Comming soon
```

This project is built upon cvt2distilgpt2 and MedSAM. We would like to thank them for their great work.

