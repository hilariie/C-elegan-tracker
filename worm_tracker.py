import cv2
import numpy as np
import skimage.filters as filters
from ultralytics import YOLO
from tracker import Tracker


def worm_checker(dets, worm_count, n_init, worm_check):
    """
    Checks if all 6 worms have been detected consecutively, then increase the n_init parameter

    Parameters
    ----------
    dets : int
        number of detected worms
    worm_count : int
        number of times all 6 worms have been detected consecutively
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    worm_check : boolean
        True if all worms have been detected consecutively for n_init number of times, otherwise False

    Returns
    -------
    worm_count : int
        an increment or reinitialization of worm_count
    n_init : int
        an increment or same value of n_init
    worm_check : boolean
        True if all worms have been detected consecutively for n_init number of times, otherwise False
    """
    if dets == 6:
        worm_count += 1
    else:
        worm_count = 0
    if worm_count == n_init + 1:
        n_init = 400
        worm_check = False
    return worm_count, n_init, worm_check


model = YOLO('models/coloured_detection_n.pt')
image_segmentation = False
video_path = 'data/videos/mating6.wmv'
if image_segmentation:
    output_path = f"results/mating_gray_tracking.mp4"
output_path = f"results/mating_tracking.mp4"

# Read video
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
# skip the part where they set up the worms
cap.set(cv2.CAP_PROP_POS_FRAMES, 6867)  # 920)
ret, frame = cap.read()
H, W = frame.shape[:2]
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# set variables to be used in video processing
tracker = Tracker()
track_dist = {}
objects = {}
colors = {}
trajectory = {}
contacts = []

vid_frames = 0
colour_black = 0
worm_count = 0
frame_count = 0
n_init = 10
worm_check = True
if image_segmentation:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = filters.threshold_local(gray_frame, block_size=101, offset=10)
while ret:
    frame_count += 1
    if image_segmentation:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = gray_frame < threshold
        gray_frame = np.where(gray_frame, 255, 0).astype(np.uint8)
        # Set all three channels to be identical to the grayscale image
        gray_frame = np.stack((gray_frame,) * 3, axis=-1)
        frame = gray_frame
    # get predictions
    results = model(frame)[0]
    # list to store predicted bbox and scores of each worm
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        label = results.names[int(class_id)]
        # if contact is predicted with a good confidence, plot bbox and move to next iteration
        if label == 'contact':
            if score > 0.78:
                cv2.rectangle(frame, (int(x1) - 5, int(y1) - 5), (int(x2) + 5, int(y2) + 5), colour_black, 2)
                cv2.putText(frame, 'contact', (int(x1) - 5, int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_black, 1,
                            cv2.LINE_AA)

                time_elapsed = frame_count / fps
                minutes = int(time_elapsed / 60)
                seconds = int(time_elapsed % 60)
                contacts.append(f"{minutes:02d}:{seconds:02d}")
            continue

        detections.append([x1, y1, x2, y2, score, class_id])
    # check if worm_check is True, so we do not perform this operation all the time.
    if worm_check:
        worm_count, n_init, worm_check = worm_checker(len(detections), worm_count, n_init, worm_check)

    tracker.update(frame, detections, n_init)

    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = map(int, bbox)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        track_id = track.track_id
        class_id = track.class_id

        if track_id not in trajectory:
            trajectory[track_id] = [(x_center, y_center)]
        if track_id not in colors:
            colors[track_id] = (250, 180, 180)
        if track_id not in objects:
            objects[track_id] = results.names[int(class_id)]

        if vid_frames == 20:
            # calculate the distance between current location and initial location of worms
            x_center0, y_center0 = trajectory[track_id][0]
            dx = abs(x_center - x_center0) ** 2
            dy = abs(y_center - y_center0) ** 2
            eud = (dx + dy) ** 0.5
            track_dist[track_id] = eud
        # male worms are usually more mobile than female worms therefore,
        # in the next frame, we check the list of distances and get the worm that has moved the most (i.e, fastest worm)
        elif vid_frames == 21:
            # get object that has moved the most within the first 21 frames of worm detections
            max_id = max(track_dist, key=lambda k: track_dist[k])
            # we identify the fastest worm as the male worm, and others as female worms
            if track_id == max_id:
                objects[track_id] = 'male worm'
                colors[track_id] = (220, 80, 250)
            else:
                objects[track_id] = 'female worm'

        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[track_id], 2)
        cv2.circle(frame, (x_center, y_center), 3, colors[track_id], -1)
        cv2.putText(frame, objects[track_id], (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[track_id], 1,
                    cv2.LINE_AA)
        # to show the travel path of worms, we get a list of its location and draw a line connecting all points
        trajectory[track_id].append((x_center, y_center))
        if objects[track_id] == 'male worm':
            for i in range(1, len(trajectory[track_id])):
                prev_center = trajectory[track_id][i - 1]
                next_center = trajectory[track_id][i]
                cv2.line(frame, (int(prev_center[0]), int(prev_center[1])),
                         (int(next_center[0]), int(next_center[1])), colors[track_id], 2)

    if vid_frames <= 21:
        vid_frames += 1

    cv2.imshow('C-elegan worm tracker', frame)
    cap_out.write(frame)
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

# write time of contact to text file
contacts = set(contacts)
contacts = ','.join(contacts).replace(',', '\n')
with open("contact_time.txt", "w") as contacts_out:
    contacts_out.write(contacts)

cap.release()
cap_out.release()
cv2.destroyAllWindows()
