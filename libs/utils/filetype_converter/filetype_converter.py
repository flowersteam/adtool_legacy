import PIL
import imageio.v3 as iio


def is_mp4(blob: bytes) -> bool:
    try:
        meta = iio.immeta(blob, plugin="pyav")
        return meta["codec"] == "h264"
    except OSError:
        # returned if the file format is not supported
        return False


def convert(blob: bytes):
    pass
