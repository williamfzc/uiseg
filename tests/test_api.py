from uiseg import UISeg
import pathlib
import cv2

def test_api():
    pic = pathlib.Path(__file__).parent / "demo.png"
    s = UISeg()

    locations_file = s.process_image_file(pic.as_posix(), show=False)
    assert locations_file

    img = cv2.imread(pic.as_posix())
    locations_img = s.process_image(img, show=False)
    assert locations_img
