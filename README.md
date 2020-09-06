# Real-time-Video-Mosaic 

<p align="center">
 <img src="Data/demo.gif" width="623" height="354">
</p>

Demo at:
https://www.youtube.com/watch?v=on_sG_X79oQ

Link to the paper:
http://ieeexplore.ieee.org/document/7886813/

________________________________

There are both python3 and C++ code available. The C++ code has more features and sanity checks but the opencv version is obsolete. I recommend starting with python version. Since the steps are same as c++, the documentation of c++ still applies to the python version with some minor changes.

<b>Python3:</b>
- opencv verion 4.4.0 (No cuda needed)

```
  sudo apt-get install python3-opencv
  python main.py
``` 
This code reads from a video and outputs an image, i.e.`mosaic.jpg`

___________________________________________________

<b>C++:</b>
- opencv 2.4 with cuda support

C++ verison can work with live camera/webcam or a recorded video. You can read the manual to know how to feed a video stream to it or use a live camera. For live camera, it may not be robust. The output is related to quality of the camera and not having blured frames when moving.

To run in your own machine:
You need to compile CUDA for your opencv 2.411 version and then recompile the code in your machine.
There is a manual I wrote: "how to compile the code.pdf" or you can use other helpful links over the internet about compiling opencv for cuda.
I used visual studio but you can use g++ to compile but you need to link the libraries.

_______________________________________________________________________________

_______________________________________________________________________________

This project has been done under supervision of: Dr. Hadi Moradi

Advanced Robotics and Intelligent Systems

University of Tehran, Iran
____________________


If you find any content of this project/paper useful for your research, please consider citing our paper:
```
@inproceedings{amiri2016real,
  title={Real-time video stabilization and mosaicking for monitoring and surveillance},
  author={Amiri, Ali Jahani and Moradi, Hadi},
  booktitle={Robotics and Mechatronics (ICROM), 2016 4th International Conference on},
  pages={613--618},
  year={2016},
  organization={IEEE}
}
```
Thank you.

Ali Jahani Amiri
