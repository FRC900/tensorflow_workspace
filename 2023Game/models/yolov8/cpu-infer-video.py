import argparse
import cv2
from ultralytics import YOLO

WRITE_OUTPUT_VIDEO = False

def main(args: argparse.Namespace) -> None:
    # Load the YOLOv8 model
    model = YOLO(args.model)

    # Open the video file
    cap = cv2.VideoCapture(args.input_video)

    if args.save_path is not None:
        # Get input vid resolution by sampling first frame from video, then resetting back to frame 0
        _, frame = cap.read()
        _ = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print (frame.shape)

        vid_writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*"FMP4"), 30., (frame.shape[1]/2, frame.shape[0]/2))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            # Add agnostic_nms=True to remove overlapping
            # detections of different classes for a single
            # object (useful for apriltags)
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            if args.save_path is not None:
                vid_writer.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model file')
    parser.add_argument('--input-video', type=str, help='Input video')
    parser.add_argument('--save-path', type=str, default=None, help='Save annotated video to this file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)