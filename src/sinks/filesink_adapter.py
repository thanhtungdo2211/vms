"""FilesinkAdapter - Video file recording sink for DeepStream"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink
from src.common import get_nvvidconv_props, get_encoder_element, detect_platform


class FilesinkAdapter(BaseSink):
    """
    Record to AVI/MP4 with H.264 encoding.
    Pipeline: queue -> nvvideoconvert -> capsfilter -> encoder -> h264parse -> muxer -> filesink
    
    Auto-detects hardware encoder (nvv4l2h264enc), falls back to x264enc.
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
        enc_factory, enc_props = get_encoder_element(self.bitrate)

        # nvvideoconvert is always required to convert NVMM→CPU memory
        # (DeepStream elements output NVMM buffers; software encoders can't read NVMM)
        # - nvv4l2h264enc (HW): output caps keep memory:NVMM
        # - x264enc (SW): output caps WITHOUT memory:NVMM → nvvideoconvert
        #   downloads to CPU RAM, avoiding a second NVMM pool allocation
        #   (critical on Jetson where CMA is shared and limited)
        if enc_factory == "nvv4l2h264enc":
            caps_string = "video/x-raw(memory:NVMM),format=NV12"
        else:
            caps_string = "video/x-raw,format=I420"

        chain = [
            self._make("queue", f"{p}_q", {"max-size-buffers": 4, "leaky": 2}),
            self._make("nvvideoconvert", f"{p}_nv", get_nvvidconv_props()),
            self._make("capsfilter", f"{p}_caps", {
                "caps": Gst.Caps.from_string(caps_string)
            }),
            self._make(enc_factory, f"{p}_enc", enc_props),
            self._make("h264parse", f"{p}_parse", {"config-interval": -1}),
            self._make(
                "avimux" if not self.location.endswith(".mp4") else "mp4mux",
                f"{p}_mux"
            ),
            self._make("filesink", f"{p}_sink", {
                "location": self.location,
                "sync": False,
                "async": False
            }),
        ]

        for elem in chain:
            pipeline.add(elem)
        for i in range(len(chain) - 1):
            if not chain[i].link(chain[i + 1]):
                raise RuntimeError(f"Failed to link {chain[i].get_name()} -> {chain[i+1].get_name()}")

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