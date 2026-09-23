#include <metal_stdlib>
using namespace metal;

// Solid color matching tess/draw oracles: RGBA(0.4, 0.267, 0.133, 1) → BGRA ≈ 0x22,0x44,0x66,0xff
fragment float4 mesh_fragment() {
    return float4(0.4, 0.267, 0.133, 1.0);
}
