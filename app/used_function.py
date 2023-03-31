import cv2
import numpy as np

def image_processing(img, right = False):
    image_array = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_array = cv2.equalizeHist(image_array)
    image_array = image_array/255.
    image_array = image_array.astype('float32')
    image_array = np.resize(image_array, (227, 227))
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    if right:
        image_array = np.fliplr(image_array)
    image_array = image_array.reshape((3,227,227))
    image_array = np.array(image_array)
    image_array = image_array.astype('float32')
    return image_array


def draw_arrow(gaze, image, eye_pos):
    try:
        length = 100
        dx = -round(length*np.sin(gaze[0]))
        dy = round(length*np.sin(np.abs(gaze[1])))
        cv2.arrowedLine(image, tuple(np.round(eye_pos[:2]).astype(np.int32)),
                        tuple(np.round([eye_pos[0]+dx, eye_pos[1]+dy]).astype(np.int32)),
                        color = (0, 255, 0),tipLength=0.4 , thickness = 1)
    except Exception as e:
        # print(e)
        # print('blink')
        pass


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), 1)
        return (cx, cy)
    except Exception as e:
        # print(e)
        # print('blink')
        pass

def eye_pupil_from_center(x1,y1,x2,y2,eye_center):
    xc,yc = eye_center
    center = ((x1+x2)//2,(y1+y2)//2)
    r1 = np.sqrt(np.square(xc-center[0])+np.square(yc-center[1]))
    if center[0]==xc and center[1] == y1:
        theta1 = 0.0
    elif center[0]==xc:
        if yc<center[1]:
            theta1 = 270.0
        else:
            theta1 = 90.0
    elif center[1] == yc:
        if xc<center[0]:
            theta1 = 180.0
        else:
            theta1 = 0.0
    else:
        theta1 = np.rad2deg(np.arctan((yc-center[1])/(xc-center[0])))
        if xc<center[0]:
            theta1 = 180+theta1
        elif yc<center[1] and xc>center[0]:
            theta1 = 360+theta1
    return r1, theta1


def eye_direction(r, theta):
    if r <= 3:
        direction = "center"
    else:
        if theta<=90 or theta>270:
            direction = "Right"
        # elif theta>45 and theta<=135:
            # direction = "bottom"
        elif theta>90 and theta<=270:
            direction = "Left"
        # elif theta>225 and theta<=315:
            # direction = "top"
        else:
            direction = "unknown"
    return direction


def looking_direction(r_l, r_r, right_direction, left_direction):
    if abs(r_l-3)>abs(r_r-3):
        look_direction = left_direction
    else:
        look_direction = right_direction
    return look_direction

def direction_look(left_eye, right_eye):
    xl1, yl1, xlc, ylc, xl2, yl2 = left_eye
    xr1, yr1, xrc, yrc, xr2, yr2 = right_eye
    r_l, theta_l = eye_pupil_from_center(xl1,yl1,xl2,yl2,xlc,ylc)
    left_direction = eye_direction(r_l, theta_l)
    r_r, theta_r = eye_pupil_from_center(xr1,yr1,xr2,yr2,xrc,yrc)
    right_direction = eye_direction(r_r, theta_r)
    direction_looked = looking_direction(r_l, r_r, right_direction, left_direction)
    return direction_looked

def vector_to_angle(vector):
    x, y, z = vector
    theta = np.arcsin(-y)
    phi = np.arctan2(-x, -z)
    return [theta, phi]

def angle_to_vector(gaze_angel):
    theta, phi = gaze_angel
    sin = np.sin([np.deg2rad(theta), np.deg2rad(phi)])
    cos = np.cos([np.deg2rad(theta), np.deg2rad(phi)])
    x = np.multiply(cos[0], sin[1])
    y = sin[0]
    z = np.multiply(cos[0], cos[1])
    return [x, y, z]