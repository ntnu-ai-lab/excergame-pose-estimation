# excergame-pose-estimation

The openpose folder represents the folder where your openpose installation lies 
(the folder that contains build/ cmake/ and so on). The python scripts that folder 
must be run from said openpose installation folder.

To create and train CNN models Tensorflow and CUDA is required. 

The dataset directory structure is as follows

```
-root
    -Par1
    -Par2
        -1
        -2
            -Par2_Mov1_Iter1_Cam1.mp4
            ...
            -Par2_Mov1_Iter1_Cam5.mp4
            -Par2_Mov1_Iter2_Cam1.mp4
            ...
            -Par2_Mov1_Iter10_Cam5.mp4
            -Par2_Mov2_Iter1_Cam1.mp4
            ...
            -Par2_Mov17_Iter10_Cam5.mp4
        -3
        ...
        -17
    -Par3
    ...
```