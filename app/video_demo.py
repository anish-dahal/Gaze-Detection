import dlib
import cv2
import numpy as np
import torch
from model.alexnet import AlexNet
from Dataloader.dataloader import data_loader

import used_function as fun


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
kernel = np.ones((5, 5), np.uint8)


def load_model(model_path):
    global model
    model = torch.load(model_path, map_location=device)
    model.eval()


def video_gaze_prediction(input_video_path, output_video_path):
    '''reads video and performs model operations in each frame

    Parameters
    ----------
    input_video_path : str
        path for the input video
    output_video_path : str
        path for the output video
    '''
    cap = cv2.VideoCapture(input_video_path)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(w), int(h))
    number_of_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        exit(0)

    result = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"MJPG"), 10, size
    )

    # Read until video is completed
    frame_count = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            # gaze detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            for face in faces:
                shape = predictor(gray, face)
                shape = fun.shape_to_np(shape)
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                mask = fun.eye_on_mask(mask, left, shape)
                mask = fun.eye_on_mask(mask, right, shape)
                mask = cv2.dilate(mask, kernel, 5)
                eyes = cv2.bitwise_and(img, img, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]

                mid = (shape[42][0] + shape[39][0]) // 2
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)

                thresh = cv2.erode(thresh, None, iterations=2)  # 1
                thresh = cv2.dilate(thresh, None, iterations=4)  # 2
                thresh = cv2.medianBlur(thresh, 3)  # 3
                thresh = cv2.bitwise_not(thresh)

                left_eye = img[
                    shape[17, 1] : shape[39, 1] + 25, shape[17, 0] : shape[39, 0] + 25
                ]
                right_eye = img[
                    shape[22, 1] : shape[45, 1] + 25, shape[22, 0] : shape[45, 0] + 25
                ]
                
                left_center = fun.contouring(thresh[:, 0:mid], mid, img)
                right_center = fun.contouring(thresh[:, mid:], mid, img, True)

            # direction
            try:
                r_l, theta_l = fun.eye_pupil_from_center(
                    shape[36, 0], shape[36, 1], shape[39, 0], shape[39, 1], left_center
                )
                left_direction = fun.eye_direction(r_l, theta_l)
                r_r, theta_r = fun.eye_pupil_from_center(
                    shape[45, 0], shape[45, 1], shape[42, 0], shape[42, 1], right_center
                )
                right_direction = fun.eye_direction(r_r, theta_r)
                direction_looked = fun.looking_direction(
                    r_l, r_r, right_direction, left_direction
                )
                
                cv2.putText(
                    img,
                    direction_looked,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (210, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            except Exception as e:
                pass

            processed_left = fun.image_processing(left_eye)
            processed_right = fun.image_processing(right_eye, right=True)
            image = [processed_left.tolist(), processed_right.tolist()]
            image = np.array(image)
            image = image.astype("float32")
            image = torch.from_numpy(image.copy()).to(device)
            dataloader = data_loader(image, batch_size=128)
            for x in dataloader:
                with torch.no_grad():
                    y_theta, y_phi = model(x)

            left_gaze_angle = y_theta[0].item(), y_phi[0].item()
            left_gaze_vector = fun.angle_to_vector(left_gaze_angle)
            left_gaze_angle = fun.vector_to_angle(
                (left_gaze_vector[0], left_gaze_vector[1], left_gaze_vector[2])
            )
            
            right_gaze_angle = y_theta[1].item(), y_phi[1].item()
            right_gaze_vector = fun.angle_to_vector(right_gaze_angle)
            right_gaze_angle = fun.vector_to_angle(
                (right_gaze_vector[0], -right_gaze_vector[1], right_gaze_vector[2])
            )
            fun.draw_arrow(left_gaze_angle, img, left_center)
            fun.draw_arrow(right_gaze_angle, img, right_center)
            result.write(img)
            frame_count += 1
            print(f"{frame_count}/{number_of_frame}")
            
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_model("./model/saved_model/dahalanishID/best_alexnet_model.pt")
    video_gaze_prediction("../video/input/video.mp4", "../video/output/video.avi")
