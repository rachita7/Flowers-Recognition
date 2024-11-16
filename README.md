# 🌸 Flower Recognition with Oxford Flowers 102 Dataset  

## Overview  
Flower recognition is a significant research area in **Computer Vision**, involving fine-grained image classification where subtle differences between flower categories pose a challenge. This project tackles these challenges using advanced techniques, achieving **test accuracies above 95%**.  

Key difficulties include:  
- **Fine-grained classification**: Minimal visible differences between flower categories.  
- **Dataset challenges**: Limited examples per class (10 per category for training and validation, with 6,149 test images).  
- **Variability in images**: Light, scale, and pose variations, along with high intra-class variance and inter-class similarity.  

This repository uses the **Oxford Flowers 102 dataset** to explore and implement state-of-the-art approaches for robust flower recognition.  

---

## Dataset  
The **Oxford Flowers 102** dataset contains:  
- 102 flower categories.  
- 10 training and validation images per category.  
- 6,149 test images in total.  

---

## Techniques Used  
To overcome the challenges, we implemented:  
1. **Transfer Learning**: Leveraging pre-trained models to enhance learning with limited data.  
2. **Few-Shot Learning**: Training the model to generalize well from a small number of examples.  
3. **Siamese Networks with Triplet Loss**: Differentiating between categories by learning relative feature embeddings.  
4. **Vision Transformers (ViTs)**: Applying transformer architectures for improved image recognition performance.  

---

## Results  
We observe that Prototypical Networks give the best results among the few-shot learning algorithms and vision transformers record the best performance on this task of image classification as a whole.
