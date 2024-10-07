# Tree-Branch-Modeling 🌳

## Members and contact info 👥

* Hannia Ashley Alvarado Galván. -> haash2706@gmail.com
* José Ángel López Gutiérrez. -> jalg030129@gmail.com
* Edgar Leonardo García Zavala. -> leo261102leo@gmail.com

## Affiliation 🏫
This is a final project for the 2025-1 Modeling and Simulation course developed at the Universidad Nacional Autónoma de México (UNAM), at the Escuela Nacional de Estudios Superiores Unidad Morela (ENES - Morelia).This project is carried out by students of the Bachelor's Degree in Technologies for Information in the Sciences.

## Introduction
In nature, trees exhibit complex and fascinating growth patterns, especially in how their branches extend and branch out. These patterns are not random but follow specific mathematical and biological rules that can be modeled and understood through algorithms. This project aims to reproduce the Growth of a tree, and understand tools can be applied to model the growth of the tree and make it beautyful.

Through this project, we aim to graphically reproduce the branching process of trees, analyzing how different growth rules influence the final structure of a tree, both visually and numerically, allowing a deeper understanding of how plants interact with space and time during their development. This approach will enable us not only to visualize the inherent beauty in nature's patterns but also to delve into the connection between biology and mathematics in plant growth.
  
## General objective 🎯
The main goal is to reproduce the growth of a tree, with a specific focus on its branches, using an algorithmic model described on a GitHub Repository refered in References. 

## Particular objectives 🎯
* understand thd model of tree branches of a pine: This is the essential first step of the project, aimed at understanding how to model branch behavior. The focus will be on the rules that control branch growth and how these rules can be modified to generate different branching patterns.

* We'll use a repoaitory with a model to understand how it works. And analize it,

* Create a graphical visualization of the model: An important part of this project is to create a visual representation of the mathematical models developed. Using computer graphics tools, we can visualize how tree branches grow over time, allowing observation of complex growth patterns. Trying to use python.
  
* Using the algorithm on which we are based, we seek to adapt the model coefficients in such a way that it is a little more realistic, for example adapting it to the shape and thickness of the trunk, in the same way adjusting the distribution angles of the branches and the number of branches.

## Methodology 🧪

1. Graphical visualization of growth of a tree, and understarnd how branches grow based on numeric splines to simulate the tree.
2. Change the randomness of the algorithm to make it specific.

    To observe the tree's development dynamically, computer graphics tools will be used to visualize how the branches grow and expand over time.

  
3. Results analysis

    A visual and numerical analysis of the different growth patterns obtained will be carried out. The results will be compared with characteristics observed in real tree types, evaluating how changes in the branch structure.

    Variables such as how higher the grade of the spline needs to be to get an  actual representation.

4. Model validation

    To verify the accuracy of the model, the results obtained will be compared with examples of real tree growth or with models already validated in the literature.

    Model parameters will be adjusted based on these results to improve realism and precision.

### variables and functions: 

height_range


diam_range


split_height_range


split_prob


num_branches


min_can_height


max_can_width


max_can_width_height


tree_top_dist


tree_mid_dist


foliage_noise

   


### Technologies
NumPy

trimesh

laspy

matplotlib

scipy


## References 📝

* https://github.com/mitchbryson/SimpleSynthTree.git
* M. Bryson, F. Wang and J. Allworth, "Using synthetic tree data in deep learning-based individual tree segmentation using LiDAR point clouds", Remote Sens. 2023, 15(9), 2380.

https://www.mdpi.com/2072-4292/15/9/2380
