import cv2
from pathlib import Path
import numpy as np


class VideMosaic:
    def __init__(self, first_image, output_height_times=2, output_width_times=3, detector="orb"):

        if detector == "orb":
            self.detector = cv2.ORB_create(700)

        self.visualize = True

        self.process_first_frame(first_image)

        self.output_img = np.zeros(shape=(int(output_height_times * first_image.shape[0]), int(
            output_width_times*first_image.shape[1]), first_image.shape[2]))

        # offset
        self.w_offset = int(self.output_img.shape[0]/2 - first_image.shape[0]/2)
        self.h_offset = int(self.output_img.shape[1]/2 - first_image.shape[1]/2)

        self.output_img[self.w_offset:self.w_offset+first_image.shape[0],
                        self.h_offset:self.h_offset+first_image.shape[1], :] = first_image

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.H_old = np.eye(3)
        self.H_old[0, 2] = self.h_offset
        self.H_old[1, 2] = self.w_offset

    def process_first_frame(self, first_image):
        self.frame_prev = first_image
        frame_gray_prev = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.kp_prev, self.des_prev = self.detector.detectAndCompute(frame_gray_prev, None)

    def match(self, des_cur, des_prev):
        # matching
        matches = self.bf.match(des_cur, des_prev)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        if self.visualize:
            match_img = cv2.drawMatches(self.frame_cur, self.kp_cur, self.frame_prev, self.kp_prev, matches[:10], None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow('matches', match_img)
        return matches

    def filter_good_matches(self):
        pass

    def process_frame(self, frame_cur):
        self.frame_cur = frame_cur
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        self.kp_cur, self.des_cur = self.detector.detectAndCompute(frame_gray_cur, None)

        self.matches = self.match(self.des_cur, self.des_prev)

        self.H = self.findHomography(self.kp_cur, self.kp_prev, self.matches)
        self.H = np.matmul(self.H_old, self.H)

        self.warp(self.frame_cur, self.H)

        self.make_mask(self.frame_cur, self.output_img, self.H)

        self.H_old = self.H
        self.kp_prev = self.kp_cur
        self.des_prev = self.des_cur
        self.frame_prev = self.frame_cur

    @staticmethod
    def findHomography(image_1_kp, image_2_kp, matches):
        # taken from https://github.com/cmcguinness/focusstack/blob/master/FocusStack.py

        image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
        for i in range(0, len(matches)):
            image_1_points[i] = image_1_kp[matches[i].queryIdx].pt
            image_2_points[i] = image_2_kp[matches[i].trainIdx].pt

        homography, mask = cv2.findHomography(
            image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0)

        return homography

    def warp(self, frame_cur, H):
        warped_img = cv2.warpPerspective(
            frame_cur, H, (self.output_img.shape[1], self.output_img.shape[0]), flags=cv2.INTER_LINEAR)
        mask, transformed_corners = self.make_mask(frame_cur, warped_img, H)
        warped_img = self.draw_corners(warped_img, transformed_corners)

        self.output_img[warped_img > 0] = warped_img[warped_img > 0]
        output_temp = np.copy(self.output_img)
        output_temp = self.draw_corners(output_temp, transformed_corners, color=(0, 0, 255))

        cv2.imshow('output',  output_temp/255.)

        return self.output_img

    @staticmethod
    def make_mask(frame_cur, output, H):
        corner1 = np.array([0, 0])
        corner2 = np.array([frame_cur.shape[1], 0])
        corner3 = np.array([frame_cur.shape[1], frame_cur.shape[0]])
        corner4 = np.array([0, frame_cur.shape[0]])

        corners = np.array([[corner1, corner2, corner3, corner4]], dtype=np.float32)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = np.array(transformed_corners, dtype=np.int32)
        mask = np.zeros(shape=(output.shape[0], output.shape[1], 1))
        cv2.fillPoly(mask, transformed_corners, color=(1, 0, 0))
        cv2.imshow('mask', mask)

        return mask, transformed_corners

    def draw_corners(self, image, corners, color=(0, 0, 0)):
        for i in range(corners.shape[1]-1, -1, -1):
            cv2.line(image, tuple(corners[0, i, :]), tuple(
                corners[0, i-1, :]), thickness=5, color=color)
        return image


def main():
    cap = cv2.VideoCapture('Data/zoom.mjpeg')

    detector = cv2.ORB_create(700)
    # detector = cv2.SIFT_create()
    ret = False

    while ret == False:
        ret, first_frame = cap.read()
        if not ret:
            continue
        video_mosaic = VideMosaic(first_frame)
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            break
        video_mosaic.process_frame(frame_cur)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
