"""Processor Registry - Auto-discovery for BranchProcessors."""

from abc import ABC, abstractmethod
from typing import Any, Callable
import importlib
import os

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

from src.sinks.base_sink import BaseSink


class BranchProcessor(ABC):
    """Abstract base class for branch processors.

    Processors are instantiated with:
        __init__(config: dict, sink: BaseSink, source_mapper: SourceIDMapper)

    The source_mapper provides camera_id <-> source_id mapping for probes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_probes(self) -> dict[str, Callable]:
        pass

    def on_pipeline_built(self, pipeline: Gst.Pipeline, branch_info: Any) -> None:
        pass

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass


class ProcessorRegistry:
    """Global registry for BranchProcessor classes."""

    _registry: dict[str, type] = {}
    _imported: list[str] = []

    @classmethod
    def register(cls, branch_name: str):
        """Decorator to register a processor class."""
        def decorator(proc_cls: type):
            cls._registry[branch_name] = proc_cls
            print(f"[ProcessorRegistry] Registered: {branch_name} -> {proc_cls.__name__}")
            return proc_cls
        return decorator

    @classmethod
    def get_classes_for_config(cls, config: dict) -> dict[str, type]:
        """Get processor classes for configured branches (not instances).

        Returns dict of {branch_name: processor_class} for instantiation later.
        """
        classes = {}
        for name in config.get("pipeline", {}).get("branches", {}):
            if name in cls._registry:
                classes[name] = cls._registry[name]
                print(f"[ProcessorRegistry] Found: {cls._registry[name].__name__} for '{name}'")
        return classes

    @classmethod
    def auto_import(cls, apps_dir: str = "apps") -> None:
        """Auto-import processor modules from apps directory."""
        if not os.path.isdir(apps_dir):
            return

        for app in os.listdir(apps_dir):
            path = os.path.join(apps_dir, app, "processor.py")
            if os.path.isfile(path):
                module = f"apps.{app}.processor"
                if module not in cls._imported:
                    try:
                        importlib.import_module(module)
                        cls._imported.append(module)
                        print(f"[ProcessorRegistry] Auto-imported: {module}")
                    except Exception as e:
                        print(f"[ProcessorRegistry] Failed to import {module}: {e}")

    @classmethod
    def get(cls, name: str) -> type | None:
        return cls._registry.get(name)

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
        cls._imported.clear()
