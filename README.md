<div align='center'>
<h2 align="center"> Spot-Compose: A Framework for Open-Vocabulary Object Retrieval and
Drawer Manipulation in Point Clouds </h2>
<h3 align="center">Under Review</h3>

<a href="https://oliver-lemke.github.io/">Oliver Lemke</a><sup>1</sup>, <a href="https://zuriabauer.com/">Zuria Bauer</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=feJr7REAAAAJ&hl=en">René Zurbrügg</a><sup>1</sup>, <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>, <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>1</sup>, <a href="https://hermannblum.net/">Hermann Blum</a><sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft Mixed Reality & AI Labs

Spot-Compose presents a comprehensive framework for integration of modern machine perception techniques with Spot, showing experiments with object grasping and dynamic drawer manipulation.


![teaser](https://spot-compose.github.io/static/images/teaser.png)


</div>

[[Project Webpage](https://spot-compose.github.io/)]
[[Paper]() (coming soon!)]


## News :newspaper:

* **Coming soon**: release on arXiv.
* **13. March 2024**: Code released.

## Code Structure :clapper:



```
spot-compose/
├── source/                            # All source code
│   ├── utils/                         # General utility functions
│   │   ├── coordinates.py             # Coordinate calculations (poses, translations, etc.)
│   │   ├── docker_communication.py    # Communication with docker servers
│   │   ├── environment.py             # API keys, env variables
│   │   ├── files.py                   # File system handling
│   │   ├── graspnet_interface.py      # Communication with graspnet server
│   │   ├── importer.py                # Config-based importing
│   │   ├── mask3D_interface.py        # Handling of Mask3D instance segmentation
│   │   ├── point_clouds.py            # Point cloud computations
│   │   ├── recursive_config.py        # Recursive configuration files
│   │   ├── scannet_200_labels.py      # Scannet200 labels (for Mask3D)
│   │   ├── singletons.py              # Singletons for global unique access
│   │   ├── user_input.py              # Handle user input
│   │   ├── vis.py                     # Handle visualizations
│   │   ├── vitpose_interface.py       # Handle communications with VitPose docker server
│   │   └── zero_shot_object_detection.py # Object detections from images
│   ├── robot_utils/                   # Utility functions specific to spot functionality
│   │   ├── base.py                    # Framework and wrapper for all scripts
│   │   ├── basic_movements.py         # Basic robot commands (moving body / arm, stowing, etc.)
│   │   ├── advanced_movements.py      # Advanced robot commands (planning, complex movements)
│   │   ├── frame_transformer.py       # Simplified transformation between frames of reference
│   │   ├── video.py                   # Handle actions that require access to robot cameras
│   │   └── graph_nav.py               # Handle actions that require access to GraphNav service
│   └── scripts/
│       ├── my_robot_scripts/
│       │   ├── estop_nogui.py         # E-Stop
│       │   └── ...                    # Other action scripts
│       └── point_cloud_scripts/
│           ├── extract_point_cloud.py # Extract point cloud from Boston Dynamics autowalk
│           ├── full_align.py          # Align autowalk and scanned point cloud
│           └── vis_ply_point_clouds_with_coordinates.py # Visualize aligned point cloud
├── data/
│   ├── autowalk/                      # Raw autowalk data
│   ├── point_clouds/                  # Extracted point clouds from autowalks
│   ├── prescans/                      # Raw prescan data
│   ├── aligned_point_clouds/          # Prescan point clouds aligned with extracted autowalk clouds
│   └── masked/                        # Mask3D output given aligned point clouds
├── configs/                           # configs
│   ├── base.yaml                      # Uppermost level of recursive configurations
│   └── user.yaml                      # User-level configurations
├── shells/
│   ├── estop.sh                       # E-Stop script
│   ├── mac_routing.sh                 # Set up networking on workstation Mac
│   ├── ubuntu_routing.sh              # Set up networking on workstation Ubuntu
│   ├── robot_routing.sh               # Set up networking on NUC
│   └── start.sh                       # Convenient script execution
├── README.md                          # Project documentation
├── requirements.txt                   # pip requirements file
├── pyproject.toml                     # Formatter and linter specs
└── LICENSE
```

### Dependencies :memo:

The main dependencies of the project are the following:
```yaml
python: 3.8
```
You can set up a pip environment as follows :
```bash
git clone --recurse-submodules git@github.com:oliver-lemke/spot-compose.git
cd spot-compose
virtualenv --python="/usr/bin/python3.8" "venv/"
source venv/bin/activate
pip install -r requirements.txt
```

### Downloads :droplet:
The pre-trained model weigts for Yolov-based drawer detection is available [here](https://drive.google.com/file/d/11axGmSgb3zmUtq541hH2TCZ54DwTEiWi/view?usp=drive_link).

### Docker Containers :whale:

|      Name       |                                                                                   Link                                                                                   |                             Run Command                              |               Start Command               |
|:---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|:-----------------------------------------:|
|    AnyGrasp     | [craiden/graspnet:v1.0](https://hub.docker.com/layers/craiden/graspnet/v1.0/images/sha256-ec5663ce991415a51c34c00f2ea6f8ab9303a88e6ac27d418df2193c6ab40707?context=repo) |  ```docker run -p 5000:5000 --gpus all -it craiden/graspnet:v1.0```  |           ```python3 app.py```            |
|   OpenMask3D    | [craiden/openmask:v1.0](https://hub.docker.com/layers/craiden/openmask/v1.0/images/sha256-023e04ebecbfeb62729352a577edc41a7f12dc4ce780bfa8b8e81eb54ffe77f7?context=repo) |  ```docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0```  |           ```python3 app.py```            |
|     ViTPose     |  [craiden/vitpose:v1.0](https://hub.docker.com/layers/craiden/vitpose/v1.0/images/sha256-43a702300a0fffa2fb51fd3e0a6a8d703256ed2d507ac0ba6ec1563b7aee6ee7?context=repo)  |  ```docker run -p 5002:5002 --gpus all -it craiden/vitpose:v1.0```   | ```easy_ViTPose/venv/bin/python app.py``` |
| DrawerDetection | [craiden/yolodrawer:v1.0](https://hub.docker.com/layers/craiden/yolodrawer/v1.0/images/sha256-2b0e99d77dab40eb6839571efec9789d6c0a25040fbb5c944a804697e73408fb?context=repo) | ```docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0``` |           ```python3 app.py```            |





## Benchmark :chart_with_upwards_trend:
We provide detailed results and comparisons here.

### 3D Scene Graph Alignment (Node Matching)
|                    Method                     | Mean Reciprocal Rank | Hits@1 | Hits@2 | Hits@3 | Hits@4 | Hits@5 |
|:---------------------------------------------:|:--------------------:|:------:|:------:|:------:|:------:|:------:|
|  [EVA](https://github.com/cambridgeltl/eva)   |        0.867         | 0.790  | 0.884  | 0.938  | 0.963  | 0.977  | 
|                 $\mathcal{P}$                 |        0.884         | 0.835  | 0.886  | 0.921  | 0.938  | 0.951  |
|         $\mathcal{P}$ + $\mathcal{S}$         |        0.897         | 0.852  | 0.899  | 0.931  | 0.945  | 0.955  |
| $\mathcal{P}$ + $\mathcal{S}$ + $\mathcal{R}$ |        0.911         | 0.861  | 0.916  | 0.947  | 0.961  | 0.970  |
|                   SGAligner                   |        0.950         | 0.923  | 0.957  | 0.974  | 0.9823 | 0.987  |

### 3D Point Cloud Registration
| Method | CD | RRE | RTE | FMR | RR |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [GeoTr](https://github.com/qinzheng93/GeoTransformer) | 0.02247	| 1.813 | 2.79 | 98.94 | 98.49 |
| Ours, K=1 | 0.01677 | 1.425 | 2.88 | 99.85 | 98.79 |
| Ours, K=2 | 0.01111 | 1.012 | 1.67 | 99.85 | 99.40 |
| Ours, K=3 | 0.01525 | 1.736 | 2.55 | 99.85 | 98.81 | 

## TODO :soon:
- [X] ~~Add 3D Point Cloud Mosaicking~~
- [X] ~~Add Support For [EVA](https://github.com/cambridgeltl/eva)~~
- [ ] Add usage on Predicted Scene Graphs
- [ ] Add scene graph alignment of local 3D scenes to prior 3D maps
- [ ] Add overlapping scene finder with a traditional retrieval method (FPFH + VLAD + KNN)


## BibTeX :pray:
```
@article{sarkar2023sgaligner,
      title={SGAligner : 3D Scene Alignment with Scene Graphs}, 
      author={Sayan Deb Sarkar and Ondrej Miksik and Marc Pollefeys and Daniel Barath and Iro Armeni},
      journal={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
      year={2023}
}
```
## Acknowledgments :recycle:
In this project we use (parts of) the official implementations of the following works and thank the respective authors for open sourcing their methods: 

- [SceneGraphFusion](https://github.com/ShunChengWu/3DSSG) (3RScan Dataloader)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) (Registration)
- [MCLEA](https://github.com/lzxlin/MCLEA) (Alignment)
