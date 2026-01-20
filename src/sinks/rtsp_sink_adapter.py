"""RtspSinkAdapter - RTSP publishing sink for DeepStream"""

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink


class RtspSinkAdapter(BaseSink):
    """
    Publish stream to RTSP server (e.g. MediaMTX) using rtspclientsink.
    Pipeline: queue -> nvvideoconvert -> caps(NV12) -> nvv4l2h264enc -> h264parse -> rtspclientsink(with internal payloader)
    """

    _counter = 0

    def __init__(self, location: str, bitrate: int = 4000000, protocols: str = "tcp"):
        self.location = location
        self.bitrate = bitrate
        self.protocols = protocols
        self.elements = []
        RtspSinkAdapter._counter += 1
        self._id = RtspSinkAdapter._counter

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        p = f"rtsp{self._id}"

        # 1. Main chain (up to parser)
        # Note: rtspclientsink expects elementary stream if we provide a payloader,
        # or it can autodetect. We will explicitly provide rtph264pay to control properties.
        chain_elements = [
            self._make("queue", f"{p}_q", {"max-size-buffers": 30, "leaky": 2}),
            self._make("nvvideoconvert", f"{p}_nv", {"compute-hw": 1, "nvbuf-memory-type": 3}),
            self._make("capsfilter", f"{p}_caps", {"caps": Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")}),
            # Hardware Encoder
            self._make("nvv4l2h264enc", f"{p}_enc", {
                "bitrate": self.bitrate,
                "tuning-info-id": 2,  # LowLatencyPreset
                "iframeinterval": 30,
                "idrinterval": 30,
            }),
            # Parser required for H264
            self._make("h264parse", f"{p}_parse", {"config-interval": -1}),
        ]

        # Add chain elements to pipeline
        for elem in chain_elements:
            pipeline.add(elem)

        # Link chain elements
        for i in range(len(chain_elements) - 1):
            if not chain_elements[i].link(chain_elements[i + 1]):
                raise RuntimeError(f"Failed to link {chain_elements[i].get_name()} -> {chain_elements[i+1].get_name()}")

        # 2. RTSP Sink
        sink = self._make("rtspclientsink", f"{p}_sink", {
            "location": self.location,
            "protocols": self.protocols, # TCP is more reliable for publishing
            "latency": 100,
            "do-rtsp-keep-alive": True
        })
        pipeline.add(sink)

        # 3. Payloader (Separate)
        # Note: Do NOT add to pipeline bin directly. The sink pad takes ownership/manages it.
        # If we add it to the pipeline, GStreamer might complain about it not being linked or state managed correctly
        # if the sink is also trying to manage it.
        payloader = self._make("rtph264pay", f"{p}_pay", {
            "config-interval": 1,
            "pt": 96
        })

        # 4. Request Pad & Configure
        sink_pad = sink.get_request_pad("sink_%u")
        if not sink_pad:
            raise RuntimeError("Failed to get request pad from rtspclientsink")

        # Set the payloader element on the sink pad
        sink_pad.set_property("payloader", payloader)

        # 5. Link Parser -> Sink Pad
        parser = chain_elements[-1]
        src_pad = parser.get_static_pad("src")
        if not src_pad:
             raise RuntimeError("Failed to get src pad from parser")

        ret = src_pad.link(sink_pad)
        if ret != Gst.PadLinkReturn.OK:
             raise RuntimeError(f"Failed to link parser to rtspclientsink: {ret}")

        # Store all elements
        self.elements = chain_elements + [sink, payloader]

        # Return the first element (queue) for upstream linking
        return chain_elements[0]

    def _make(self, factory: str, name: str, props: dict = None) -> Gst.Element:
        elem = Gst.ElementFactory.make(factory, name)
        if not elem:
            # Fallback for non-Jetson environments (testing)
            if factory == "nvv4l2h264enc":
                print(f"[RtspSinkAdapter] Warning: {factory} not found, falling back to x264enc")
                return self._make_sw_enc_fallback(name, props)
            if factory == "nvvideoconvert":
                print(f"[RtspSinkAdapter] Warning: {factory} not found, falling back to videoconvert")
                return Gst.ElementFactory.make("videoconvert", name)

            raise RuntimeError(f"Cannot create element: {factory}")

        for k, v in (props or {}).items():
            if k == "protocols":
                # Convert string protocol to enum if necessary, or let GStreamer handle string
                # GstRTSPLowerTrans: 4=tcp
                if v == "tcp":
                    v = 4
                elif v == "udp":
                    v = 1
            elem.set_property(k.replace("-", "_"), v)
        return elem

    def _make_sw_enc_fallback(self, name: str, props: dict) -> Gst.Element:
        """Create x264enc bin if HW encoder is missing"""
        elem = Gst.ElementFactory.make("x264enc", name)
        elem.set_property("bitrate", (props.get("bitrate", 4000000) // 1000))
        elem.set_property("speed-preset", "ultrafast")
        elem.set_property("tune", "zerolatency")
        return elem

    def start(self) -> None:
        print(f"[RtspSinkAdapter] Publishing to: {self.location}")

    def stop(self) -> None:
        print(f"[RtspSinkAdapter] Stopped publishing to: {self.location}")
