# Joint Intensity Classification and Specimen Segmentation on HEp-2 Images: a Deep Learning Approach

Repository of the model, loss, training and testing scripts of the related work submitted to ICPR 2022.

This is the implementation of a joint approach on two important tasks of HEp-2 image analysis. The two tasks are the intensity classification and specimen segmentation, which are fundamental and required tasks for the development of an end-to-end system.

In the repository are present the training and testing (cross_validation.py), the loss (loss.py) and the model used (models.py).
There is a reduced version of the framework used as it is still under development for further work on generalization, integration of new tasks and performance improvement.
After the realization of the dataloader for the specific dataset and data type the framework could be used executing cross_validation.py.
Moreover, the cross_validation.py file accept different parameters that could be analyzed in the file with their description.

**G. Percannella, U. Petruzzello, P. Ritrovato, L. Rundo, F. Tortorella and M. Vento, "Joint Intensity Classification and Specimen Segmentation on HEp-2 Images: a Deep Learning Approach," 2022 26th International Conference on Pattern Recognition (ICPR), Montreal, QC, Canada, 2022, pp. 4343-4349, doi: 10.1109/ICPR56361.2022.9956212.**
