# pick_camera.py — preview each available camera so you can find which index
# is your iPhone (the one showing the street).
#   press  n  → next camera
#   press  q  → quit
# Whatever index is showing the street is the number to use for SOURCE.
import cv2


def main():
    idx = 0
    misses = 0
    while idx < 8 and misses < 3:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            idx += 1
            misses += 1
            continue
        misses = 0
        print(f"Showing camera index {idx} — press 'n' for next, 'q' to quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.putText(frame, f"Camera index {idx}   (n = next,  q = quit)",
                        (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 215, 255), 2)
            cv2.imshow("Pick your camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        idx += 1
    cv2.destroyAllWindows()
    print("No more cameras found.")


if __name__ == "__main__":
    main()
