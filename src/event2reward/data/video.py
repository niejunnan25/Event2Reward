"""Frame-accurate video reading with decord, falling back to PyAV."""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


class DecordFrameReader:
    """Worker-local random-access reader with layered codec fallbacks.

    TorchCodec is preferred because it can fetch a sparse group of exact frame
    indices in one native call.  This matters for progress samples, where the
    start, recent history, and goal can be hundreds of frames apart.  Decord is
    retained for compatible files and PyAV is the correctness fallback.
    """

    BACKENDS = ("torchcodec", "decord", "pyav")

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        cache_size: int = 8,
        num_threads: int = 1,
        allow_pyav_fallback: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.cache_size = int(cache_size)
        self.num_threads = int(num_threads)
        self.allow_pyav_fallback = bool(allow_pyav_fallback)
        self._readers: OrderedDict[str, object] = OrderedDict()
        self._backend: dict[str, str] = {}
        self._timestamps: dict[str, np.ndarray] = {}
        self._backend_read_calls = {name: 0 for name in self.BACKENDS}
        self._backend_frames = {name: 0 for name in self.BACKENDS}
        self._backend_open_failures = {name: 0 for name in self.BACKENDS}
        self._backend_fallbacks: dict[str, int] = {}
        if self.cache_size <= 0:
            raise ValueError("cache_size must be positive")

    def _resolve(self, relative_path: str) -> Path:
        root = Path(os.path.abspath(self.dataset_root))
        path = Path(os.path.abspath(root / relative_path))
        # Compare real locations: a mount may be reached through a symlink.
        real_root, real_path = Path(os.path.realpath(root)), Path(os.path.realpath(path))
        if real_root not in real_path.parents and real_path != real_root:
            raise ValueError(f"video path escapes dataset root: {relative_path}")
        return path

    def _reader(self, relative_path: str):
        path = str(self._resolve(relative_path))
        reader = self._readers.pop(path, None)
        if reader is None:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            reader = self._open_reader(path, self.BACKENDS)
        self._readers[path] = reader
        while len(self._readers) > self.cache_size:
            evicted_path, evicted = self._readers.popitem(last=False)
            self._close_reader(evicted)
            self._backend.pop(evicted_path, None)
            self._timestamps.pop(evicted_path, None)
        return reader

    def _open_reader(self, path: str, backends: tuple[str, ...]):
        last_error: Exception | None = None
        for backend in backends:
            if backend == "pyav" and not self.allow_pyav_fallback:
                continue
            try:
                if backend == "torchcodec":
                    from torchcodec.decoders import VideoDecoder

                    reader = VideoDecoder(path, device="cpu", seek_mode="approximate")
                elif backend == "decord":
                    from decord import VideoReader, cpu

                    reader = VideoReader(path, ctx=cpu(0), num_threads=self.num_threads)
                elif backend == "pyav":
                    import av

                    reader = av.open(path)
                    # FFmpeg's automatic decoder thread count is multiplied by
                    # every DDP rank, DataLoader worker, and cached camera.
                    # Keep the fallback's thread budget explicit.
                    reader.streams.video[0].thread_count = self.num_threads
                else:  # pragma: no cover - internal contract
                    raise ValueError(f"unknown video backend: {backend}")
            except Exception as exc:
                self._backend_open_failures[backend] += 1
                last_error = exc
                continue
            self._backend[path] = backend
            return reader
        policy = (
            "torchcodec/decord only (PyAV disabled)"
            if not self.allow_pyav_fallback
            else "torchcodec/decord/PyAV"
        )
        raise RuntimeError(
            f"no permitted video backend could open {path}; policy={policy}"
        ) from last_error

    @staticmethod
    def _close_reader(reader: object) -> None:
        close = getattr(reader, "close", None)
        if callable(close):
            close()

    def _fallback_reader(self, path: str, failed_backend: str):
        remaining = {
            "torchcodec": ("decord", "pyav"),
            "decord": ("pyav",),
        }.get(failed_backend, ())
        if not remaining:
            raise RuntimeError(f"no fallback after video backend {failed_backend}")
        failed_reader = self._readers.pop(path, None)
        if failed_reader is not None:
            self._close_reader(failed_reader)
        self._backend.pop(path, None)
        self._timestamps.pop(path, None)
        reader = self._open_reader(path, remaining)
        self._readers[path] = reader
        selected_backend = self._backend[path]
        transition = f"{failed_backend}_to_{selected_backend}"
        self._backend_fallbacks[transition] = (
            self._backend_fallbacks.get(transition, 0) + 1
        )
        return reader

    def _record_read(self, backend: str, frame_count: int) -> None:
        self._backend_read_calls[backend] += 1
        self._backend_frames[backend] += int(frame_count)

    def read(self, relative_path: str, indices: list[int] | tuple[int, ...]) -> list[Image.Image]:
        reader = self._reader(relative_path)
        if not indices:
            return []
        path = str(self._resolve(relative_path))
        normalized = [int(index) for index in indices]
        if min(normalized) < 0:
            raise IndexError(f"negative frame index: {normalized}")
        while True:
            backend = self._backend[path]
            if backend == "pyav":
                images = self._read_pyav(reader, normalized)
                self._record_read(backend, len(images))
                return images
            if backend == "torchcodec":
                count = int(reader.metadata.num_frames)
                if max(normalized) >= count:
                    raise IndexError(f"frame index outside [0,{count}): {normalized}")
                try:
                    frames = reader.get_frames_at(indices=normalized).data
                    images = [
                        Image.fromarray(
                            frame.permute(1, 2, 0).contiguous().numpy(), mode="RGB"
                        )
                        for frame in frames
                    ]
                    self._record_read(backend, len(images))
                    return images
                except Exception:
                    reader = self._fallback_reader(path, backend)
                    continue
            count = len(reader)
            if max(normalized) >= count:
                raise IndexError(f"frame index outside [0,{count}): {normalized}")
            try:
                frames = reader.get_batch(normalized).asnumpy()
                images = [
                    Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
                    for frame in frames
                ]
                self._record_read(backend, len(images))
                return images
            except Exception:
                reader = self._fallback_reader(path, backend)

    def read_timestamps(
        self,
        relative_path: str,
        timestamps_s: list[float] | tuple[float, ...],
        *,
        tolerance_s: float,
    ) -> list[Image.Image]:
        """Decode the encoded frames nearest to episode-local query times.

        Dataset frame indices and encoded video frame indices are not always
        interchangeable.  Galaxea, for example, declares a 15 FPS dataset
        timeline while its current head-camera MP4 stream is about 62 FPS.
        Every selected frame is therefore checked against the encoded PTS
        instead of silently treating a dataset frame number as an MP4 index.
        """

        queries = np.asarray(timestamps_s, dtype=np.float64)
        if queries.ndim != 1:
            raise ValueError("timestamps_s must be one-dimensional")
        if len(queries) == 0:
            return []
        if not np.isfinite(queries).all() or float(queries.min()) < 0:
            raise ValueError("timestamps_s must contain finite non-negative values")
        if tolerance_s <= 0:
            raise ValueError("tolerance_s must be positive")
        reader = self._reader(relative_path)
        path = str(self._resolve(relative_path))
        while True:
            backend = self._backend[path]
            if backend == "pyav":
                images = self._read_pyav_timestamps(reader, queries, tolerance_s)
                self._record_read(backend, len(images))
                return images
            if backend == "torchcodec":
                try:
                    average_fps = float(reader.metadata.average_fps)
                    frame_count = int(reader.metadata.num_frames)
                    selected = np.rint(queries * average_fps).astype(np.int64)
                    if int(selected.min()) < 0 or int(selected.max()) >= frame_count:
                        raise IndexError(
                            f"timestamp-derived frame index outside [0,{frame_count}): "
                            f"{selected.tolist()}"
                        )
                    frames = reader.get_frames_at(indices=selected.tolist())
                    loaded_times = (
                        frames.pts_seconds.detach().cpu().numpy().astype(np.float64)
                    )
                    distances = np.abs(loaded_times - queries)
                    if (distances > float(tolerance_s)).any():
                        raise IndexError("TorchCodec timestamp lookup exceeded tolerance")
                    images = [
                        Image.fromarray(
                            frame.permute(1, 2, 0).contiguous().numpy(), mode="RGB"
                        )
                        for frame in frames.data
                    ]
                    self._record_read(backend, len(images))
                    return images
                except Exception:
                    reader = self._fallback_reader(path, backend)
                    continue

            try:
                encoded_timestamps = self._timestamps.get(path)
                if encoded_timestamps is None:
                    count = len(reader)
                    raw = np.asarray(
                        reader.get_frame_timestamp(list(range(count))), dtype=np.float64
                    )
                    if raw.ndim == 2:
                        if raw.shape[1] < 1:
                            raise RuntimeError(f"empty Decord timestamp metadata for {path}")
                        raw = raw[:, 0]
                    encoded_timestamps = raw.reshape(-1)
                    if (
                        len(encoded_timestamps) != count
                        or not np.isfinite(encoded_timestamps).all()
                    ):
                        raise RuntimeError(f"invalid Decord timestamp metadata for {path}")
                    if np.any(np.diff(encoded_timestamps) < 0):
                        raise RuntimeError(f"non-monotonic encoded timestamps in {path}")
                    self._timestamps[path] = encoded_timestamps

                positions = np.searchsorted(encoded_timestamps, queries, side="left")
                right = np.clip(positions, 0, len(encoded_timestamps) - 1)
                left = np.clip(positions - 1, 0, len(encoded_timestamps) - 1)
                choose_left = np.abs(encoded_timestamps[left] - queries) <= np.abs(
                    encoded_timestamps[right] - queries
                )
                selected = np.where(choose_left, left, right)
                distances = np.abs(encoded_timestamps[selected] - queries)
            except Exception:
                reader = self._fallback_reader(path, backend)
                continue
            invalid = distances > float(tolerance_s)
            if invalid.any():
                details = [
                    {
                        "query_s": float(query),
                        "nearest_s": float(encoded_timestamps[index]),
                        "distance_s": float(distance),
                    }
                    for query, index, distance in zip(
                        queries[invalid], selected[invalid], distances[invalid], strict=True
                    )
                ]
                raise IndexError(
                    f"encoded frame timestamp exceeds tolerance_s={tolerance_s} "
                    f"for {path}: {details[:8]}"
                )
            try:
                frames = reader.get_batch(selected.tolist()).asnumpy()
            except Exception:
                reader = self._fallback_reader(path, backend)
                continue
            images = [
                Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
                for frame in frames
            ]
            self._record_read(backend, len(images))
            return images

    @staticmethod
    def _read_pyav(container, indices: list[int] | tuple[int, ...]) -> list[Image.Image]:
        wanted = sorted({int(index) for index in indices})
        if wanted[0] < 0:
            raise IndexError("negative frame index")
        stream = container.streams.video[0]
        declared_frame_count = int(stream.frames or 0)
        terminal_index = declared_frame_count - 1 if declared_frame_count > 0 else None
        fps = float(stream.average_rate)
        # Seek to the preceding keyframe. PTS-to-index conversion keeps dataset
        # frame indices exact for AgiBot's constant-frame-rate AV1 videos.
        start_seconds = wanted[0] / fps
        target_pts = max(0, int(start_seconds / float(stream.time_base)))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        found: dict[int, Image.Image] = {}
        last_decoded_image: Image.Image | None = None
        wanted_set = set(wanted)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_index = round(float(frame.pts * stream.time_base) * fps)
            image = frame.to_image().convert("RGB")
            last_decoded_image = image
            if frame_index in wanted_set:
                found[frame_index] = image
            # Some nominally constant-rate videos have an irregular final PTS.
            # If that PTS maps the physical last frame one slot past the
            # declared ordinal index, retain this already-decoded image as the
            # terminal frame. Do not decode to EOF: long source videos make
            # that fallback prohibitively expensive in DataLoader workers.
            terminal_requested = terminal_index is not None and terminal_index in wanted_set
            if frame_index > wanted[-1]:
                if terminal_requested and terminal_index not in found:
                    found[terminal_index] = image
                break
            if len(found) == len(wanted_set):
                break
        missing = [index for index in wanted if index not in found]
        if (
            terminal_index is not None
            and terminal_index in missing
            and last_decoded_image is not None
        ):
            found[terminal_index] = last_decoded_image
            missing.remove(terminal_index)
        if missing:
            raise IndexError(f"PyAV could not decode requested frame indices: {missing}")
        return [found[int(index)].copy() for index in indices]

    @staticmethod
    def _read_pyav_timestamps(
        container,
        timestamps_s: np.ndarray,
        tolerance_s: float,
    ) -> list[Image.Image]:
        stream = container.streams.video[0]
        first = float(timestamps_s.min())
        last = float(timestamps_s.max())
        target_pts = max(0, int(first / float(stream.time_base)))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        loaded_times: list[float] = []
        loaded_images: list[Image.Image] = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            loaded_times.append(timestamp)
            loaded_images.append(frame.to_image().convert("RGB"))
            # The first encoded frame at or after the final query is sufficient
            # to choose the nearest neighbor for every query.
            if timestamp >= last:
                break
        if not loaded_times:
            raise IndexError("PyAV decoded no timestamped frames")
        encoded = np.asarray(loaded_times, dtype=np.float64)
        selected: list[int] = []
        distances: list[float] = []
        for query in timestamps_s:
            index = int(np.abs(encoded - query).argmin())
            selected.append(index)
            distances.append(float(abs(encoded[index] - query)))
        invalid = [
            (float(query), float(encoded[index]), distance)
            for query, index, distance in zip(timestamps_s, selected, distances, strict=True)
            if distance > tolerance_s
        ]
        if invalid:
            raise IndexError(
                f"PyAV encoded frame timestamp exceeds tolerance_s={tolerance_s}: "
                f"{invalid[:8]}"
            )
        return [loaded_images[index].copy() for index in selected]
