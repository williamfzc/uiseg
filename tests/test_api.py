from uiseg import UISeg

import pathlib


def test_api():
    pic = pathlib.Path(__file__).parent / "demo.png"
    s = UISeg()
    locations = s.process_image(pic.as_posix(), show=True)
    assert locations
