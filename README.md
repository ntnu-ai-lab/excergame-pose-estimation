# excergame-pose-estimation

The repo directory named "openpose" represents the folder where your OpenPose installation lies 
(the folder that contains build/ cmake/ and so on). The python scripts in that directory 
must be run from said openpose installation folder. For information on how to install
and use OpenPose, see https://github.com/CMU-Perceptual-Computing-Lab/openpose

To create and train CNN models Tensorflow 2.7.0 and CUDA is required. 

The dataset directory structure is assumed to be structured like this:
```
-dataset
    -Par1
    -Par2
        -1
            -Par2_Mov1_Iter1_Cam5.csv
            -Par2_Mov1_Iter2_Cam5.csv
            ...
            -Par2_Mov1_Iter10_Cam5.csv
        -2
            -Par2_Mov2_Iter1_Cam5.csv
            ...
            -Par2_Mov2_Iter10_Cam5.csv
        -3
        ...
        -17
    -Par3
    ...
```
