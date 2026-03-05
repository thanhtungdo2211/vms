from gi.repository import Gst
from src.sinks.base_sink import BaseSink

class MultifilesinkAdapter(BaseSink):
    _counter = 0

    def __init__(self, location: str, max_files: int = 100, quality: int = 85):
        self.location = location
        self.max_files = max_files
        self.quality = quality
        MultifilesinkAdapter._counter += 1
        self._id = MultifilesinkAdapter._counter

    def create(self, pipeline: Gst.Pipeline) -> Gst.Element:
        p = f"mjpeg{self._id}"
        from src.common import make_element
        elems = [
            make_element("nvvideoconvert", f"{p}_conv", {"nvbuf-memory-type": 3}),
            make_element("capsfilter", f"{p}_caps", {"caps": Gst.Caps.from_string("video/x-raw,format=I420")}),
            make_element("jpegenc", f"{p}_enc", {"quality": self.quality}),
            make_element("multifilesink", f"{p}_sink", {"location": self.location, "max-files": self.max_files}),
        ]
        for e in elems:
            pipeline.add(e)
        for i in range(len(elems) - 1):
            elems[i].link(elems[i + 1])
        return elems[0]  # First element — upstream links here

    def start(self): pass
    def stop(self): pass