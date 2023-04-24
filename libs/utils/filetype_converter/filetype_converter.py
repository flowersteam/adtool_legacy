import PIL
import imageio.v3 as iio
from io import BytesIO


def is_mp4(blob: bytes) -> bool:
    try:
        meta = iio.immeta(blob, plugin="pyav")
        return meta["codec"] == "h264"
    except OSError:
        # returned if the file format is not supported
        return False


def convert_from_image(blob: bytes) -> bytes:
    ndarray = iio.imread(blob)
    bbuf = BytesIO()
    iio.imwrite(uri=bbuf, image=ndarray, extension=".mp4")
    return bbuf.getvalue()


def convert_from_video(blob: bytes) -> bytes:
    # there's actually no impl difference because we just call imageio,
    # although there is a short circuit if the file is already an mp4
    if is_mp4(blob):
        return blob
    else:
        return convert_from_image(blob)
