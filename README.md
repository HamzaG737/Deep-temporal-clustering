# Deep-temporal-clustering pytorch implemention
A non-official pytorch implementation of the DTC model , presented in the paper :
> Madiraju, N. S., Sadat, S. M., Fisher, D., & Karimabadi, H. (2018). Deep Temporal Clustering : Fully Unsupervised Learning of Time-Domain Features. http://arxiv.org/abs/1802.01059

This an unsupervised architecture for the classification of multivariate time series. 

## Usage 
To train the model , you can run the following command : 
```shell
$ python3 train.py --similarity --pool
```
Note that the similarity and pool arguments are required. To see the full list of arguments , including the dataset name,  please refer to the *config.py* file. 

The autoencoder and clustering models weights will be saved in a **models_weights** directory. Also the train.py file returns the ROC score corresponding to the training parameters. 

## Further improvements  
* Add heatmap network 
* Add Auto Correlation based Similarity. 
* Output more metrics for training. 
