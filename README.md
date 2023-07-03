<div id="header" align="center">
  <h1>
    Worm tracker with OpenCV
    <img src="results\worm_tracker.jpg" alt="worm tracker screenshot" width="300" align="center"/>
  </h1>
</div>

This project aims to detect and track a male C-elegan worm's mating behaviour amongst female worms. It uses Yolov8 by ultralytics to detect the worms,
 uses DeepSort algorithm to track the worms, detects when contact between a male and female worm has been made, and then it records the contact time for
scientific research.

## Dependencies

To run this project, you need to install the following dependency:

- `ultralytics`
- `tensorflow`
- `scikit-image`
- `filterpy`
- `numpy`


## How to Use

1. Clone the repository using the command: `git clone <repo_url>`
2. Navigate to the project directory: `cd <repo>`
3. Edit the path to the input worm video in the `worm_tracker` script.
4. Run the script using the command: `python worm_tracker.py`

#### For segmentation.py
1. if segmentation is set to True, after running the script, a gray scale image will appear asking if the segmentation performed on the image is satisfactory.
2. Hit any key and on the terminal, you will be asked if you wish to adjust the parameters to make the segmentation better.This is an iterative process until you are content with the segmentation.

After this, the script will display the output frames of the video with the tracked worms. If you want to end the script/video, press 'q'. The output video will be saved as 'results/worm_tracker.mp4'.

## Results and Areas of concern
Although this approach performs significantly better than using opencv's tracking algorithms alone, there is still need for further improvements particularly in the detection aspect.
