"""Sample still frames from a video, for models that take images but not video."""

import base64 as _base64
import io
import logging
from typing import TYPE_CHECKING, List, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from flo_ai.models.chat_message import VideoMessageContent

# Long side of a sampled frame, in pixels. Frames are sent as images, and a
# full-resolution 4K frame costs far more tokens than it adds.
MAX_FRAME_SIDE = 768


def video_to_bytes(video: 'VideoMessageContent') -> bytes:
    if video.bytes:
        return video.bytes
    if video.base64:
        try:
            return _base64.b64decode(video.base64, validate=True)
        except ValueError as exc:
            raise ValueError(
                f'VideoMessageContent.base64 is not valid base64: {exc}'
            ) from exc
    if video.url:
        raise ValueError(
            'URL-based videos are only supported by providers with native video '
            'input; fetch the video bytes first.'
        )
    raise ValueError('VideoMessageContent has no bytes, base64, or url')


def sample_video_frames(data: bytes, max_frames: int) -> List[Tuple[float, bytes]]:
    """Up to ``max_frames`` PNG frames spread evenly over the video.

    Returns ``(seconds_from_start, png_bytes)`` pairs in time order.
    """
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "Sending video to this model needs PyAV: pip install 'flo-ai[video]'"
        ) from exc
    import pymupdf

    if max_frames < 1:
        raise ValueError('max_frames must be at least 1')

    with av.open(io.BytesIO(data)) as container:
        if not container.streams.video:
            raise ValueError('Video has no video stream')
        stream = container.streams.video[0]
        time_base = stream.time_base
        start = float(stream.start_time * time_base) if stream.start_time else 0.0
        if stream.duration:
            duration = float(stream.duration * time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = 0.0

        if duration <= 0:
            # WebM recordings made by MediaRecorder commonly omit both duration
            # fields. Demux packet timestamps to estimate the end, then decode
            # sequentially; seeking such a file often returns only its first
            # frame because it has no cue table.
            last_pts = None
            for packet in container.demux(stream):
                if packet.pts is not None and (last_pts is None or packet.pts > last_pts):
                    last_pts = packet.pts
            if last_pts is not None and time_base is not None:
                duration = max(0.0, float(last_pts * time_base) - start)

            if duration > 0:
                logger.warning(
                    'Video duration metadata is missing; sampling frames by sequential decode'
                )
            else:
                logger.warning(
                    'Video duration and packet timestamps are unavailable; sampling the first decoded frame'
                )
            return _sample_durationless_frames(data, max_frames, start, duration, av, pymupdf)

        # Midpoints of equal slices, so a 1-frame sample is the middle of the
        # video rather than its (often black) first frame.
        targets = [duration * (i + 0.5) / max_frames for i in range(max_frames)]
        frames: List[Tuple[float, bytes]] = []
        last_time = None
        for target in targets:
            container.seek(int((start + target) / time_base), stream=stream)
            chosen = None
            for frame in container.decode(stream):
                chosen = frame
                if frame.time is None or frame.time + 1e-6 >= start + target:
                    break
            # Past the last frame, the final frame stands in, unless it was
            # already the previous pick.
            if chosen is None or (
                last_time is not None
                and chosen.time is not None
                and chosen.time <= last_time
            ):
                continue
            last_time = chosen.time
            frames.append((target, _frame_to_png(chosen, pymupdf)))
        if not frames:
            raise ValueError('Could not decode any frames from the video')
        return frames


def _sample_durationless_frames(
    data: bytes,
    max_frames: int,
    start: float,
    duration: float,
    av,
    pymupdf,
) -> List[Tuple[float, bytes]]:
    """Choose frames nearest evenly spaced targets without relying on seek cues."""
    targets = (
        [duration * (i + 0.5) / max_frames for i in range(max_frames)]
        if duration > 0
        else [0.0]
    )
    frames: List[Tuple[float, bytes]] = []
    target_index = 0
    previous = None
    first_untimed = None
    last_selected_time = None

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame_time = frame.time
            if frame_time is None and frame.pts is not None and stream.time_base is not None:
                frame_time = float(frame.pts * stream.time_base)
            if frame_time is None:
                # Some damaged or unusual streams expose no timestamps at all.
                # Preserve the old one-frame fallback rather than failing the video.
                if first_untimed is None:
                    first_untimed = frame
                continue

            while (
                target_index < len(targets)
                and frame_time + 1e-6 >= start + targets[target_index]
            ):
                target_time = start + targets[target_index]
                chosen = (
                    (frame_time, frame)
                    if previous is None
                    or abs(frame_time - target_time) < abs(previous[0] - target_time)
                    else previous
                )
                if last_selected_time is None or chosen[0] > last_selected_time:
                    frames.append((targets[target_index], _frame_to_png(chosen[1], pymupdf)))
                    last_selected_time = chosen[0]
                target_index += 1
            previous = (frame_time, frame)

        # If the last target lies just beyond the final timestamp, the final
        # frame stands in, matching the duration-known path's end behavior.
        if target_index < len(targets) and previous is not None:
            for target in targets[target_index:]:
                if last_selected_time is None or previous[0] > last_selected_time:
                    frames.append((target, _frame_to_png(previous[1], pymupdf)))
                    last_selected_time = previous[0]

    if not frames and first_untimed is not None:
        frames.append((0.0, _frame_to_png(first_untimed, pymupdf)))
    if not frames:
        raise ValueError('Could not decode any frames from the video')
    return frames


def _frame_to_png(frame, pymupdf) -> bytes:
    scale = min(1.0, MAX_FRAME_SIDE / max(frame.width, frame.height))
    width = max(2, int(frame.width * scale)) // 2 * 2
    height = max(2, int(frame.height * scale)) // 2 * 2
    rgb = frame.reformat(width=width, height=height, format='rgb24')
    plane = rgb.planes[0]
    raw = bytes(plane)
    row = width * 3
    if plane.line_size != row:
        raw = b''.join(
            raw[r * plane.line_size : r * plane.line_size + row] for r in range(height)
        )
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, raw, False).tobytes('png')
