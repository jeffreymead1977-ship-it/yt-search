"""Extract panoramic images embedded in an E57 file and prepare them for sphere rendering.

E57 binary format:
  - 48-byte file header: signature(8) + major(u32) + minor(u32) +
    filePhysLen(u64) + xmlPhysOffset(u64) + xmlLogLen(u64) + pageSize(u64)
  - Pages are 1024 bytes (1020 data + 4-byte CRC32c at end of each page).
  - XML lives at xmlPhysOffset.
  - Blob elements: fileOffset + length — NOT page-formatted; raw read.
"""
from __future__ import annotations

import io
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pyvista as pv
    _PYVISTA_OK = True
except ImportError:
    _PYVISTA_OK = False


@dataclass
class PanoramaData:
    scan_index:   int
    scanner_pos:  np.ndarray     # (3,) float64
    image:        np.ndarray     # (H, W, 3) uint8
    image_width:  int
    image_height: int


_HEADER_FMT = "<8sIIQQQQ"   # 8+4+4+8+8+8+8 = 48 bytes
_PAGE_SIZE   = 1024
_PAGE_DATA   = 1020          # bytes of data per page (last 4 = CRC)


def _read_xml(f, xml_phys_offset: int, xml_log_len: int) -> str:
    """Read the XML section from an open E57 file handle."""
    f.seek(xml_phys_offset)
    data = bytearray()
    while len(data) < xml_log_len:
        page = f.read(_PAGE_SIZE)
        if not page:
            break
        data.extend(page[:_PAGE_DATA])
    return data[:xml_log_len].decode("utf-8", errors="replace")


def _read_blob(f, file_offset: int, length: int) -> bytes:
    f.seek(file_offset)
    return f.read(length)


def _ns_tag(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _find_named(parent: ET.Element, name: str) -> Optional[str]:
    for ch in parent:
        if ch.get("name") == name:
            return ch.text
    return None


def _find_translation(elem: ET.Element) -> np.ndarray:
    """Look for pose/translation x,y,z under elem."""
    for child in elem.iter():
        if _ns_tag(child.tag) == "translation":
            try:
                x = float(_find_named(child, "x") or 0)
                y = float(_find_named(child, "y") or 0)
                z = float(_find_named(child, "z") or 0)
                return np.array([x, y, z], dtype=np.float64)
            except Exception:
                pass
    return np.zeros(3, dtype=np.float64)


def _decode_image(data: bytes) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.uint8)
    except Exception:
        return None


def extract_panoramas(path: str) -> list[PanoramaData]:
    """Extract all panoramic images from an E57 file. Returns [] on any failure."""
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(48)
            if len(header_bytes) < 48:
                return []
            (sig, major, minor,
             file_phys_len, xml_phys_offset,
             xml_log_len, page_size) = struct.unpack(_HEADER_FMT, header_bytes)

            xml_text = _read_xml(f, xml_phys_offset, xml_log_len)
            try:
                root_el = ET.fromstring(xml_text)
            except ET.ParseError:
                return []

            results: list[PanoramaData] = []
            scan_index = 0

            _IMAGE_TAGS  = {"jpegImage", "pngImage"}
            _SPHERE_TAGS = {"sphericalRepresentation", "equirectangularRepresentation"}

            def _iter_elem(elem: ET.Element):
                return [(ch, _ns_tag(ch.tag)) for ch in elem]

            def _search_images(data3d_list):
                nonlocal scan_index
                for scan_el in data3d_list:
                    pos = _find_translation(scan_el)
                    images2d = None
                    for ch, tag in _iter_elem(scan_el):
                        if tag == "images2D":
                            images2d = ch
                            break
                    if images2d is None:
                        scan_index += 1
                        continue

                    for img_el, _img_tag in _iter_elem(images2d):
                        repr_el = None
                        for ch, tag in _iter_elem(img_el):
                            if tag in _SPHERE_TAGS:
                                repr_el = ch
                                break
                        target = repr_el if repr_el is not None else img_el

                        blob_el = None
                        for ch, tag in _iter_elem(target):
                            if tag in _IMAGE_TAGS:
                                blob_el = ch
                                break
                        if blob_el is None:
                            for ch in target.iter():
                                if ch.get("fileOffset") is not None:
                                    blob_el = ch
                                    break

                        if blob_el is None:
                            continue

                        try:
                            file_offset = int(blob_el.get("fileOffset", -1))
                            length      = int(blob_el.get("length", 0))
                            if file_offset < 0 or length <= 0:
                                continue
                        except (TypeError, ValueError):
                            continue

                        raw = _read_blob(f, file_offset, length)
                        img_arr = _decode_image(raw)
                        if img_arr is None:
                            continue

                        results.append(PanoramaData(
                            scan_index=scan_index,
                            scanner_pos=pos,
                            image=img_arr,
                            image_width=img_arr.shape[1],
                            image_height=img_arr.shape[0],
                        ))

                    scan_index += 1

            data3d_el = None
            for el in root_el.iter():
                if _ns_tag(el.tag) == "data3D":
                    data3d_el = el
                    break

            if data3d_el is not None:
                _search_images(list(data3d_el))
            else:
                # Fallback: look for any element with jpegImage/pngImage recursively
                for el in root_el.iter():
                    if _ns_tag(el.tag) in _IMAGE_TAGS:
                        try:
                            file_offset = int(el.get("fileOffset", -1))
                            length      = int(el.get("length", 0))
                            if file_offset < 0 or length <= 0:
                                continue
                            raw = _read_blob(f, file_offset, length)
                            img_arr = _decode_image(raw)
                            if img_arr is None:
                                continue
                            results.append(PanoramaData(
                                scan_index=len(results),
                                scanner_pos=np.zeros(3),
                                image=img_arr,
                                image_width=img_arr.shape[1],
                                image_height=img_arr.shape[0],
                            ))
                        except Exception:
                            continue

            return results

    except Exception:
        return []


def make_panorama_sphere(pano: PanoramaData, radius: float = 80.0):
    """Return (pv.PolyData sphere, pv.Texture) ready for plotter.add_mesh."""
    if not _PYVISTA_OK:
        raise RuntimeError("pyvista not available")
    sphere = pv.Sphere(
        radius=radius,
        center=tuple(float(v) for v in pano.scanner_pos),
        theta_resolution=180,
        phi_resolution=90,
    )
    sphere.flip_normals()
    texture = pv.Texture(pano.image)
    return sphere, texture
