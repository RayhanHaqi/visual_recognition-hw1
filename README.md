# Visual Recognition - HW1: 100-Class Image Classification
This repository contains the implementation for Homework 1 of the Visual Recognition using Deep Learning course at National Yang Ming Chiao Tung University (NYCU). The project focuses on high-accuracy image classification across 100 classes using a Knowledge Distillation (KD) framework.

## Introduction
The core methodology of this project is Sequential Knowledge Distillation. To achieve high performance on a 100-class dataset while maintaining an efficient inference model, a powerful "Teacher" model (such as SigLIP or ConvNeXt XXLarge) was used to generate soft labels. These labels were then used to train a "Student" model (ResNet-RS-200), allowing the student to learn complex feature representations from the teacher.
