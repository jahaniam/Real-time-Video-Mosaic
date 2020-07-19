import cv2
from pathlib import Path


def main():
    cap = cv2.VideoCapture('Data/rotate.mjpeg')

    orb = cv2.ORB_create(500)
    ret = False

    while ret == False:
        ret, frame_prev = cap.read()
        if not ret:
            continue

        frame_gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        kp_prev, des_prev = orb.detectAndCompute(frame_gray_prev, None)

    while cap.isOpened():
        ret, frame_cur = cap.read()
        if not ret:
            continue
        frame_gray_cur = cv2.cvtColor(frame_cur, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame_cur)


        # feature detection
        kp_cur, des_cur = orb.detectAndCompute(frame_gray_cur, None)

        # matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_cur, des_prev)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches( frame_cur, kp_cur,frame_prev, kp_prev, matches[:10], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('matches',img3)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
