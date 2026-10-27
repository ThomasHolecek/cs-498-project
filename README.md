Gesture Recognition to Action:
	Will Thompson, Joe Thompson, Thomas Holecek

Code (see RECORDING.md for the full how-to)
	record_gestures.py: guided recording sessions that build the gesture dataset in data/
	review_clips.py: play back recorded clips, trim each to just the gesture, reject or relabel bad ones
	train_gestures.py: trains the come_here / shoo / stop / none classifier on data/
	live_gestures.py: runs the classifier live on a webcam or the K1 camera stream, can save video
	k1_camera_stream.py: runs on the K1, streams its head camera to the laptop over HTTP

Overview
For this project we want to develop a gesture controlled follow the person system. When a person does a “come here” hand gesture the robot will lock on to that person and try to maintain a short distance between them. If the person is farther away it will approach them using a larger “come here” gesture. If the person walks away the robot will follow them. While following them the robot turns its head to keep the person in view so as  to not lose tracking of them. If the person does a “stop” gesture by opening their palm at the robot, will stop walking but still be locked on to that person. A second “come here” gesture will resume the following. And if the person does a “shooing” gesture the robot will unlock from the person and take a few steps back.
Some benchmarks that we can measure are how well it can track someone walking. How smooth it can stop when shown the stop gesture. We could also measure the percent of the time it correctly identifies each gesture for the stop, follow, and shoo gestures.
The ideal deliverable by the end of the semester is a working demonstration of this.
Related Work
Following on Wheeled Robot:
[F. Rollo, A. Zunino, G. Raiola, F. Amadio, A. Ajoudani, and N. Tsagarakis, “FollowMe: A robust person following framework based on visual re-identification and gestures,” in Proc. IEEE Int. Conf. Adv. Robot. Social Impacts (ARSO), 2023, pp. 84–89, doi: 10.1109/ARSO56563.2023.10187536.
Gesture Recognition:
C. Lugaresi et al., “MediaPipe: A framework for building perception pipelines,” arXiv:1906.08172, 2019.
Following:
P. Nikdel, R. Vaughan, and M. Chen, “LBGP: Learning based goal planning for autonomous following in front,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2021, pp. 3140–3146.
Simulation:
NVIDIA, M. Mittal et al., “Isaac Lab: A GPU-accelerated simulation framework for multi-modal robot learning,” arXiv:2511.04831, 2025.
Data Requirements
The model that we are going to use, MediaPipe, has some gestures pretrained into the model, we will make our own gestures. To do this we will record our own dataset of gestures and train a small classifier on the hand and body points MediaPipe detects. We will need a few hundred examples per gesture stored as small CSV files. This will mainly be recording of each gesture at distances of 1-4 meters from the camera at several different angles
We will need to simulate people to walk in Isaac lab so that the robot can learn to follow them.
Compute Requirements
For gesture recognition we will most likely need 0-1 GPU-hours. For the base following policy we are estimating 15-30 compute hours, in order to generalize on variable follow speeds, start distances we estimate another 10-18 hours. In total we are estimating 60 hours, with 40- 60GB of storage. Any additional compute time required past the allocated amount to our team we can offload to our personal compute power (Colab Pro, RTX 3080, M5 CPU).

Timeline
Gesture Recognition Oct 12 - Oct 30:
Oct 12-16: Record Gestures and train classifier to add Gestures to MediaPipe
Use MediaPipe to detect hand and body movements
Create dataset for come here, stop, and shoo
Oct 19-30 re record through the k1’s camera and angle at 1-4m and retrain
Goal: recognize all three gestures on a laptop
Head tracking Oct 19 - Oct 29:
Detect the person’s body using MediaPipe pose
Control robot’s head to track the person
Goal: robot can detect a person and moves its head to keep them in view
Weekly Stand up Oct 29th:
Presenting: 
Current progress: Most likely Gesture Recognition and Head Tracking
What we have learned
Difficulties
What were planning to do next: Detection and Following
Detection and Following Nov 2 - Nov 19:
Use built in walking policy to move
Lock onto the person after the come here gesture, tracking position and distance
Train following policy
Goal: Robot can maintain a person’s location relative to itself
Weekly Stand up Nov 19th:
Presenting: 
Progress since last update: Most likely Detection and Following
What we have learned since the last stand up
Difficulties we have had since the last stand up
What we will do next: Full Gesture Recognition to Action
Full gesture-recognition to action Nov 19 - Dec 4:
 Combine gesture recognition, person tracking, and following
Add smooth stop/start
Test gesture accuracy, following accuracy, and overall functionality
Goal: working end of semester demonstration 

By the final presentation we hope to have the robot be able to follow a person, recognize gestures, and smoothly follow them with its head.


