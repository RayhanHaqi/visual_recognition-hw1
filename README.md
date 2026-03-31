# Visual Recognition - HW1: 100-Class Image Classification

This repository contains the implementation for Homework 1 of the Visual Recognition using Deep Learning course at National Yang Ming Chiao Tung University (NYCU). The project focuses on high-accuracy image classification across 100 classes using a Knowledge Distillation (KD) framework.

## Introduction

The core methodology of this project is Sequential Knowledge Distillation. To achieve high performance on a 100-class dataset while maintaining an efficient inference model, a powerful "Teacher" model (such as SigLIP or ConvNeXt XXLarge) was used to generate soft labels. These labels were then used to train a "Student" model (ResNet-RS-200), allowing the student to learn complex feature representations from the teacher.

## Environment Setup

To set up the environment, it is recommended to use a Python virtual environment or Conda.

1. **Clone the repository:**
   
   ```
   git clone https://github.com/RayhanHaqi/visual_recognition-hw1.git
   cd visual_recognition-hw1
   ```
   
2. **Install dependencies:**
   The project requires PyTorch, Torchvision, and other common deep learning libraries.
   
   ```
   pip install torch torchvision torchaudio
   pip install timm tqdm matplotlib pandas
   ```
   
3. **Hardware Requirement:**
   Due to the size of the Teacher models, an NVIDIA GPU with at least 16GB of VRAM (e.g., RTX 3080/4080 or higher) is recommended for the labeling phase.

## Usage

* To train the student models, edit the train-auto-full.py to change paths variables and the names of models to train.
  
   ```
   MODELS = ['resnetrs200.tf_in1k', 'resnest200e.in1k']
   DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
   ```
   
   ```
   python train-auto-full.py
   ```

* To train the student models with full tweaking and 448 px resolution input layer, edit the train-auto-insane.py to change paths variables and the names of models to train.

  
   ```
   MODELS = ['resnetrs200.tf_in1k', 'resnest200e.in1k']
   DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
   ```
   
   ```
   python train-auto-insane.py
   ```

* To train the teacher models, edit the train-teacher.py to change paths variables and the names of models to train.

   ```
   MODELS = [
    'vit_so400m_patch14_siglip_378.webli_ft_in1k', 
    'convnext_xxlarge.clip_laion2b_soup_ft_in1k', 
    'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
   ]
   DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
   ```
   
   ```
   python train-teacher.py
   ```
   
* To do the distillation knowledge learning, edit the train-distill.py to change paths variables and the names of models to train

   ```
   TEACHER_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/vit_so400m_patch14_siglip_378.webli_ft_in1k_teacher_best.pth'
   TEACHER_MODEL_NAME = 'vit_so400m_patch14_siglip_378.webli_ft_in1k'
   
   STUDENT_MODEL_NAME = 'resnetrs200.tf_in1k'
   STUDENT_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/insane_resnetrs200.tf_in1k_run8_best.pth' 
   
   DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data'
   ```
   
   ```
   python train-distill.py
   ```

* To generate submission file for codabench, edit the generate-submission.py to change paths variables and the names of models to train

   ```
   CHECKPOINT_PATH = '/home/tilakoid/selectedtopics/cv_hw1_data/checkpoints/distill_resnetrs200.tf_in1k-vit_so400m_patch14_siglip_378.pth'
   
   DATA_DIR = '/home/tilakoid/selectedtopics/cv_hw1_data/data/' 
   ```

   ```
   python generate-submission.py
   ```
  

## Performance Snapshot
The implementation achieves the following performance metrics on the 100-class classification task:

* Student Architecture: ResNet-RS-200

* Teacher Architecture: SigLIP / ConvNeXt XXLarge

* Top-1 Accuracy: 0.96+

