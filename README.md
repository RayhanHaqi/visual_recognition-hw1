# Visual Recognition - HW1: 100-Class Image Classification
This repository contains the implementation for Homework 1 of the Visual Recognition using Deep Learning course at National Yang Ming Chiao Tung University (NYCU). The project focuses on high-accuracy image classification across 100 classes using a Knowledge Distillation (KD) framework.

## Introduction
The core methodology of this project is Sequential Knowledge Distillation. To achieve high performance on a 100-class dataset while maintaining an efficient inference model, a powerful "Teacher" model (such as SigLIP or ConvNeXt XXLarge) was used to generate soft labels. These labels were then used to train a "Student" model (ResNet-RS-200), allowing the student to learn complex feature representations from the teacher.

## Environment Setup
To set up the environment, it is recommended to use a Python virtual environment or Conda.
1. **Clone the repository:**
   'git clone https://github.com/RayhanHaqi/visual_recognition-hw1.git'
   'cd visual_recognition-hw1'
2. **Install dependencies:**
   The project requires PyTorch, Torchvision, and other common deep learning libraries.
   'pip install torch torchvision torchaudio'
   'pip install timm tqdm matplotlib pandas'
3. **Hardware Requirement:**
   Due to the size of the Teacher models, an NVIDIA GPU with at least 16GB of VRAM (e.g., RTX 3080/4080 or higher) is recommended for the labeling phase.
