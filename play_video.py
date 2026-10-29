"""
PRADAS — Smooth Video Player (Normal 1.0x Real-Time Speed).

Plays annotated videos at exact 1.0x real-time framerate.
Controls:
    - Space: Pause / Play
    - Right Arrow / D: Forward 2 seconds
    - Left Arrow / A: Rewind 2 seconds
    - Q / ESC: Quit
"""

import sys
import os
import time
import argparse
import cv2


def play(video_path: str, speed: float = 1.0) -> None:
    """Play video at exact real-time playback speed."""
    if not os.path.exists(video_path):
        print(f"Erreur: Fichier vidéo introuvable : {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo : {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = 1.0 / (fps * speed)

    window_title = f"PRADAS - Lecture Vitesse Normale 1.0x ({fps:.1f} FPS)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720)

    print("=" * 60)
    print(f"Lecture en cours : {video_path}")
    print(f"Vitesse : {speed}x ({fps:.1f} FPS) | Total : {total_frames} frames")
    print("Contrôles :")
    print("  [Espace] : Pause / Reprendre")
    print("  [D / Flèche Droite] : +2 secondes")
    print("  [A / Flèche Gauche] : -2 secondes")
    print("  [Q / Échap] : Quitter")
    print("=" * 60)

    paused = False

    while cap.isOpened():
        if not paused:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Fin de la vidéo.")
                break

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.imshow(window_title, frame)

            elapsed = time.time() - start_time
            sleep_ms = max(1, int((delay - elapsed) * 1000))
        else:
            sleep_ms = 30

        key = cv2.waitKey(sleep_ms) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('d') or key == 83:  # D or Right
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, cur + int(fps * 2)))
        elif key == ord('a') or key == 81:  # A or Left
            cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - int(fps * 2)))

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="PRADAS Smooth 1x Video Player")
    parser.add_argument(
        "--video", type=str, default="data/output/mot13_crowdhuman_annotated.mp4",
        help="Chemin de la vidéo à lire"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Vitesse de lecture (1.0 = normale, 1.5 = rapide, 0.5 = ralenti)"
    )
    args = parser.parse_args()
    play(args.video, args.speed)


if __name__ == "__main__":
    main()
