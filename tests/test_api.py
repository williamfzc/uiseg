from uiseg import UISeg
import pathlib
import cv2

pic = pathlib.Path(__file__).parent / "demo.png"


def test_api_file():
    s = UISeg()
    locations_file = s.process_image_file(pic.as_posix(), show=False)
    assert locations_file


def test_api():
    s = UISeg()
    img = cv2.imread(pic.as_posix())
    locations_img = s.process_image(img, show=False)
    assert locations_img
