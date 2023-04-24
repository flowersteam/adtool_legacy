from filetype_converter.filetype_converter import is_mp4, convert


def setup_function(function):
    global TEST_ARTIFACTS
    asset_folder = "filetype_converter/tests/assets"
    img_path = asset_folder + "/img.png"
    vid_path = asset_folder + "/vid.mp4"
    doc_path = asset_folder + "/doc.docx"
    with open(img_path, "rb") as f:
        img = f.read()
    with open(vid_path, "rb") as f:
        vid = f.read()
    with open(doc_path, "rb") as f:
        doc = f.read()
    TEST_ARTIFACTS = {"img": img,
                      "vid": vid,
                      "doc": doc}


def teardown_function(function):
    pass


def test_is_mp4():
    assert is_mp4(TEST_ARTIFACTS["vid"])
    assert not is_mp4(TEST_ARTIFACTS["img"])
    assert not is_mp4(TEST_ARTIFACTS["doc"])


def test_convert_image():
    pass


def test_convert_video():
    pass


def test_convert_errors():
    pass
