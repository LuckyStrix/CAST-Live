"""Write a triangle mesh as binary glTF (GLB): vertex-coloured, or textured in parts. numpy + struct only.

glTF is right-handed, Y-up, metres; COLOR_0 is linear light, so sRGB colours are decoded.
(From FleshForge's glbwrite, same author, MIT.)
"""

import json
import struct

import numpy as np

_FLOAT, _UINT32 = 5126, 5125
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class _Builder:
    """Buffers, views and accessors for one GLB."""

    def __init__(self):
        self.chunks, self.views, self.accessors = [], [], []

    def view(self, raw: bytes, target=None) -> int:
        offset = sum(len(c) for c in self.chunks)
        self.chunks.append(raw + b"\x00" * (-len(raw) % 4))
        v = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw)}
        if target:
            v["target"] = target
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, data, target, comp, kind, minmax=False) -> int:
        acc = {"bufferView": self.view(data.tobytes(), target), "componentType": comp,
               "count": len(data), "type": kind}
        if minmax:
            acc["min"], acc["max"] = data.min(axis=0).tolist(), data.max(axis=0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def write(self, path, doc: dict) -> None:
        binary = b"".join(self.chunks)
        doc.update({"asset": {"version": "2.0", "generator": "CAST Live"},
                    "buffers": [{"byteLength": len(binary)}], "bufferViews": self.views,
                    "accessors": self.accessors})
        text = json.dumps(doc, separators=(",", ":")).encode()
        text += b" " * (-len(text) % 4)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + 8 + len(binary)))
            fh.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
            fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def _check(vertices, faces):
    vertices = np.ascontiguousarray(vertices, dtype="<f4")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty (V, 3) array")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.max() >= len(vertices):
        raise ValueError("faces must be a non-empty (F, 3) array of valid vertex indices")
    return vertices, faces


def _material(name, texture=None):
    pbr = {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0, "roughnessFactor": 1.0}
    if texture is not None:
        pbr["baseColorTexture"] = {"index": texture}
    return {"name": name, "doubleSided": True, "pbrMetallicRoughness": pbr}


def write_glb(path, vertices, faces, colors_srgb=None, normals=None, name="scan"):
    """One vertex-coloured mesh."""
    vertices, faces = _check(vertices, faces)
    b, attributes = _Builder(), {}
    attributes["POSITION"] = b.accessor(vertices, _ARRAY_BUFFER, _FLOAT, "VEC3", minmax=True)
    if normals is not None:
        attributes["NORMAL"] = b.accessor(np.ascontiguousarray(normals, dtype="<f4"),
                                          _ARRAY_BUFFER, _FLOAT, "VEC3")
    if colors_srgb is not None:
        lin = np.clip(srgb_to_linear(np.clip(colors_srgb, 0, 1)), 0, 1)
        attributes["COLOR_0"] = b.accessor(np.ascontiguousarray(lin, dtype="<f4"),
                                           _ARRAY_BUFFER, _FLOAT, "VEC3")
    indices = b.accessor(faces.reshape(-1), _ELEMENT_ARRAY_BUFFER, _UINT32, "SCALAR")
    b.write(path, {
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [
            {"attributes": attributes, "indices": indices, "material": 0, "mode": 4}]}],
        "materials": [_material("scan")]})


def write_textured_glb(path, parts, name="scan"):
    """A mesh in parts, each `(vertices, normals, uvs, faces, image)`: `image` is the part's
    texture as PNG or JPEG bytes, or None for an untextured (grey) part. UVs have their origin
    at the image's top left, as glTF expects."""
    b, primitives, materials, images, textures = _Builder(), [], [], [], []
    for i, (vertices, normals, uvs, faces, image) in enumerate(parts):
        vertices, faces = _check(vertices, faces)
        attributes = {"POSITION": b.accessor(vertices, _ARRAY_BUFFER, _FLOAT, "VEC3", True)}
        if normals is not None:
            attributes["NORMAL"] = b.accessor(np.ascontiguousarray(normals, dtype="<f4"),
                                              _ARRAY_BUFFER, _FLOAT, "VEC3")
        texture = None
        if image is not None:
            attributes["TEXCOORD_0"] = b.accessor(np.ascontiguousarray(uvs, dtype="<f4"),
                                                  _ARRAY_BUFFER, _FLOAT, "VEC2")
            mime = "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"
            images.append({"bufferView": b.view(image), "mimeType": mime})
            textures.append({"source": len(images) - 1})
            texture = len(textures) - 1
        materials.append(_material(f"part{i}", texture))
        primitives.append({"attributes": attributes, "mode": 4, "material": len(materials) - 1,
                           "indices": b.accessor(faces.reshape(-1), _ELEMENT_ARRAY_BUFFER,
                                                 _UINT32, "SCALAR")})
    doc = {"scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
           "meshes": [{"name": name, "primitives": primitives}], "materials": materials}
    if images:
        doc.update(images=images, textures=textures)
    b.write(path, doc)
