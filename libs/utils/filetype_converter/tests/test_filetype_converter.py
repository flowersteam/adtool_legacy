from filetype_converter.filetype_converter import (is_mp4,
                                                   convert_from_image,
                                                   convert_from_video)
import imageio.v3 as iio


def setup_function(function):
    global TEST_ARTIFACTS
    asset_folder = "filetype_converter/tests/assets"
    img_path = asset_folder + "/img.png"
    vid_path = asset_folder + "/vid.mp4"
    doc_path = asset_folder + "/doc.docx"
    mov_path = asset_folder + "/mkv.mkv"
    with open(img_path, "rb") as f:
        img = f.read()
    with open(vid_path, "rb") as f:
        vid = f.read()
    with open(doc_path, "rb") as f:
        doc = f.read()
    with open(mov_path, "rb") as f:
        mkv = f.read()
    TEST_ARTIFACTS = {"img": img,
                      "vid": vid,
                      "doc": doc,
                      "mkv": mkv}


def teardown_function(function):
    pass


def test_is_mp4():
    assert is_mp4(TEST_ARTIFACTS["vid"])
    assert not is_mp4(TEST_ARTIFACTS["img"])
    assert not is_mp4(TEST_ARTIFACTS["doc"])


def test_convert_image():
    vid = convert_from_image(TEST_ARTIFACTS["img"])

    assert is_mp4(vid)

    meta = iio.immeta(vid, plugin="pyav")
    ndarray = iio.imread(vid, plugin="pyav")
    # test that the video is a single frame, two ways
    assert meta["duration"]*meta["fps"] == 1.
    assert ndarray.shape[0] == 1


def test_convert_video():
    vid = convert_from_video(TEST_ARTIFACTS["vid"])
    assert is_mp4(vid)
    # assert that the video is the same as the original
    assert vid == TEST_ARTIFACTS["vid"]

    mkv = convert_from_video(TEST_ARTIFACTS["mkv"])
    assert is_mp4(mkv)
    # assert that the video is not the same as the original
    # due to nontrivial conversion
    assert mkv != TEST_ARTIFACTS["mkv"]
