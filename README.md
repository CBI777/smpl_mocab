<h1 align="center">
SMPL_Mocab
</h1>
<h4 align="center">
3d motion capture using smpl mesh, with frankmocab + humaniflow
</h4>

-------


<p align="center">
    <img src="https://github.com/CBI777/smpl_mocab/blob/master/README_Img/demo_smpl.gif">
</p>


-------
SMPL_Mocap provides real-time 3d motion capture in Python from Ubuntu environment.
We combined real-time pose estimation model of FrankMocab with pose / shape estimation model of HuManiFlow to make real-time pose and shape estimation for video input provided via webcam of the computer.
The Project currently features : 
  - Real-time pose estimation with ~40fps
  - Concurrent shape estimation from wanted frames
  - Representing rendered SMPL Model with input video
  - Output estimated result to .obj file or file that can be converted into .obj file

---------

### News:
  - [2023/09/01] Repository commit.

---------

## Installation

We recommend you to run this project on **Ubuntu** or **Mac**. Some libraries are not compatible to **Windows**.

Tested ubuntu version : 22.04

### Clone this repository
```angular2html
git clone https://github.com/CBI777/smpl_mocab.git
cd smpl_mocap
```

### Set up an environment
- Install basic dependencies
  ```
  sudo apt-get install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev freeglut3-dev libosmesa6-dev
  sudo apt-get install ffmpeg
  ```
#### With Anaconda virtual env

- Create an Anaconda virtual environment
  - Before running command, you should modify ```smpl_mocap.yml``` file's last line, depending on your Anaconda path.
  - Run command.
    ```
    conda env create -f smpl_mocap.yml
    ```
  - Activate a virtual environment.
    ```angular2html
    conda activate smpl_mocap
    ```
- Install Cuda
  - We tested on cuda 11.7 and 12.2.
- Install pytorch with specific 'find link' option
  ```
  pip install pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
  ```
- Downgrade numpy
  - Without this step, there is an error like ```AttributeError: module 'numpy' has no attribute 'bool```.
  ```
  pip install numpy==1.23
  ```

#### Without Anaconda

- Install python>=3.8
- Install libraries from requirements.txt
```
pip install -r requirements.txt
```
- Follow extra steps after **Create an Anaconda virtual environment** mentioned above

### Download additional files
There are extra files not uploaded on this repository because of the github capacity issue.
- Please follow Installation guide of [FrankMocap](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md) and download necessary models.

### (Optional) Download additional libraries for testing original modules
Install Detectron2, pytorch3d, etc.
See detailes on [FrankMocap](https://github.com/facebookresearch/frankmocap/blob/main/docs/INSTALL.md) 
and [HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow).

---------

## A Quick Start
- Run real-time pose / shape estimation
  ```
  python -m demo.demo_rt
  ```
- While running the project, you can use some key inputs to control the project :
    - [ESC] : Exit the project
    - [Spacebar] : Estimate shape from current frame. Multiple inputs within short period of time output mean value of : the result of each shape estimation results.
    - [O(alphabet)] : Toggles output file feature. Default is {Off}, thus you should press 'o' to start file output and press it once more to stop file output so that you can store output between certain frame and frame.
  - Make sure to have the rendering window as active for key input.
  - GLRenderer window will only be active when there is something to render under current code structure. So, please have at least one person visible on webcam before using command.


- Convert output .txt file to .obj file
  ```
  python -m demo.psToObj
  ```
  - Note that default output is .txt, which stores 3x3 rotation of each SMPL joints and beta values for shape.
    - .txt file output goes into {PSOutput} folder.
    - Converted .obj file output goes into {Output} folder.
    - Make sure to empty those two folders before you start the project and turn on the output feature to prevent your output data from getting contaminated by previous results.
  
  - If you change the code so that the project outputs .obj files directly, you won't need this code.
    - In this case, .obj output goes into Output folder directly if you do not change anything with the saving directory of the code.

---------

## References

- SMPL_Mocap was benefited from the following open-sources shared in the research community :
    - [SMPL](https://smpl.is.tue.mpg.de/), [SMPLX](https://smpl-x.is.tue.mpg.de/)
    - [FrankMocap](https://github.com/facebookresearch/frankmocap)
    - [HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow)
