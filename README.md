# Non-maxima-suppression-for-object-detection-in-Cpp-and-Java
Non-maxima suppression for object detection in C++ and Java

Non-Maxima Suppression (NMS) of detection bounding boxes is one of most vital components in an object detection pipeline. There are many NMS algorithms in the Computer Vision literature with different strengths and weaknesses. One of the most successful ones is the greedy suppression approach, popularized by [1][2][3], due to its efficiency and effectiveness. The original code made available by the authors in [1] was written in MATLAB with vectorized operations to make it fast. The speed of the MATLAB code was further greatly increased by Tomasz Malisiewicz who managed to remove an inner loop and vectorize the code even further.

However, there are still no open source NMS implementations for C++ and Java. Here, I use the Armadillo C++ linear algebra library to implement the greedy NMS algorithm in C++. Additionally, I provide a Java API using the Java Native Interface (JNI) so that it can be conveniently used in any object detection systems written in Java, in addition to any object detection systems written in C++.

https://kyaw.xyz/2017/12/10/non-maxima-suppression-for-object-detection-in-c-and-java/

Copyright (C) 2017 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.

Dr. Kyaw Kyaw Htike @ Ali Abdul Ghafur

[https://kyaw-kyaw-htike.github.io](https://kyaw-kyaw-htike.github.io)

[1] A Discriminatively Trained, Multiscale, Deformable Part Model
P. Felzenszwalb, D. McAllester, D. Ramanan
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2008

[2] Object Detection with Discriminatively Trained Part Based Models
P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan
IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 32, No. 9, September 2010

[3] Object Detection Grammars
P. Felzenszwalb, D. McAllester
University of Chicago, Computer Science TR-2010-02, February 2010

Dr. Kyaw Kyaw | Kyaw Kyaw | Ali Abdul Ghafur
