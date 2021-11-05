
# Motion Forecasting with Unlikelihood Training inContinuous Space #
This repository contains the code for [Motion Forecasting with Unlikelihood Training in Continuous Space](https://openreview.net/forum?id=4u25M570Iji).
Here we applied the unlikelihood loss with [Trajectron++](https://arxiv.org/abs/2001.03093) and test on the nuScenes dataset.
Code is based on its official [implementation](https://github.com/StanfordASL/Trajectron-plus-plus).

## Installation ##


### Environment Setup ###
```
conda create --name unlike python=3.6
conda activate unlike
pip install -r requirements.txt
```

### Dataset Setup ###
A preprocessed nuScenes Dataset is provided [here](https://drive.google.com/drive/folders/1A508m8MsK2TI0U2y0brI01Xh-K6M_uAp?usp=sharing)
In case you'd like to preprocess the dataset by youself,
first download the nuScenes dataset (this requires signing up on [their website](https://www.nuscenes.org/). We use v1.0 following Trajectron++).  
Then, download the map expansion pack (v1.1) and copy the contents of the extracted `maps` folder into the `maps` folder of the dataset. 
Finally, process them into a data format that our model can work with.
We use CuPy, a cuda version of numpy, to speed up the data preprocessing. You can install it following [here](https://cupy.dev/)

```
cd experiments/nuScenes
python process_data.py --data=./v1.0 --version="v1.0" --output_path=/path/to/save/
```

## Model Training ##

To train a model on the nuScenes dataset, you can execute one of the following commands from within the `trajectron/` directory, depending on the model version you desire.

| Model                                     | Command                                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Traj.++ with L_unlike                     | `python train.py --conf ../experiments/nuScenes/models/int_ee_me/config.json  --train_data_dict /path/to/nuScenes_train_full.pkl  --eval_data_dict /path/to/nuScenes_val_full.pkl  --offline_scene_graph yes  --preprocess_workers 12  --log_dir /path/to/log_root  --train_epochs 35 --node_freq_mult_train --log_tag unlike --map_encoding  --augment`                      |
| Traj.++                                   | `python train.py --conf ../experiments/nuScenes/models/int_ee_me/config.json  --train_data_dict /path/to/nuScenes_train_full.pkl  --eval_data_dict /path/to/nuScenes_val_full.pkl  --offline_scene_graph yes  --preprocess_workers 12  --log_dir /path/to/log_root  --train_epochs 35 --node_freq_mult_train --log_tag traj++ --map_encoding  --augment  --unlike_type no`    |

## Model Evaluation ##

To evaluate a trained model's performance on forecasting vehicles, you can execute a one of the following commands from within the `experiments/nuScenes` directory.
We provide three pretrained models in `experiments/nuScenes/models/`. 
`int_ee_me\` contains the original pretrained Trajectron++ same as the one in their repository.
`trajectron_pp\` contains the pretrained Trajectron++ with our hyperparameters that lead to a better performance.
`unlikelihood\` contains the pretrained model trained with the unlikelihood loss.

`python evaluate.py --model /path/to/model/ --checkpoint the_one_you_like  --data /path/to/nuScenes_test_full.pkl  --prediction_horizon 8`


