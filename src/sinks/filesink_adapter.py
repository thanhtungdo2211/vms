"""FilesinkAdapter - Video file recording sink for DeepStream"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink
from src.common import get_nvvidconv_props, get_encoder_element, detect_platform, link_chain


class FilesinkAdapter(BaseSink):
    """
    Record to AVI/MP4 with H.264 encoding.
    Pipeline: queue -> nvvideoconvert -> RGBA -> identity -> videoconvert -> encoder -> muxer -> filesink
    """

    _counter = 0

    def __init__(self, location: str = "output.avi", bitrate: int = 4000000):
        self.location = location
        self.bitrate = bitrate
        self.elements = []
        FilesinkAdapter._counter += 1
        self._id = FilesinkAdapter._counter

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        p = f"file{self._id}"
        platform = detect_platform()

        # Get platform-optimized encoder
        enc_factory, enc_name, enc_props = get_encoder_element(p, self.bitrate)

        chain = [
            self._make("queue", f"{p}_q", {"max-size-buffers": 30, "leaky": 2}),
            self._make("nvvideoconvert", f"{p}_nv", get_nvvidconv_props()),
            self._make("capsfilter", f"{p}_caps", {"caps": Gst.Caps.from_string("video/x-raw,format=RGBA")}),
            self._make("identity", f"{p}_id", {"drop-probability": 0}),
            self._make("videoconvert", f"{p}_vc"),
            self._make("queue", f"{p}_q2", {"max-size-buffers": 30, "leaky": 2}),
            self._make(enc_factory, enc_name, enc_props),
            self._make("h264parse", f"{p}_parse", {"config-interval": -1}),
            self._make("avimux" if not self.location.endswith(".mp4") else "mp4mux", f"{p}_mux"),
            self._make("filesink", f"{p}_sink", {"location": self.location, "sync": False, "async": False}),
        ]

        for elem in chain:
            pipeline.add(elem)
        link_chain(chain)

        self.elements = chain
        print(f"[FilesinkAdapter] Using encoder: {enc_factory} ({platform.name})")
        return chain[0]

    def _make(self, factory: str, name: str, props: dict = None) -> Gst.Element:
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            raise RuntimeError(f"Cannot create element: {factory}")
        for k, v in (props or {}).items():
            elem.set_property(k.replace("-", "_"), v)
        return elem

    def start(self) -> None:
        print(f"[FilesinkAdapter] Recording to: {self.location}")

    def stop(self) -> None:
        print(f"[FilesinkAdapter] Stopped: {self.location}")
