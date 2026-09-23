#include <metal_stdlib>
using namespace metal;

struct ControlPoint {
    float4 position [[attribute(0)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> control_points;
};

struct VertexOut {
    float4 position [[position]];
};

// Post-tessellation vertex: triangle patch, 3 control points, barycentric blend.
[[patch(triangle, 3)]]
vertex VertexOut tess_vertex(
    PatchIn patchIn [[stage_in]],
    float3 patch_coord [[position_in_patch]])
{
    float4 p0 = patchIn.control_points[0].position;
    float4 p1 = patchIn.control_points[1].position;
    float4 p2 = patchIn.control_points[2].position;
    VertexOut out;
    out.position = p0 * patch_coord.x + p1 * patch_coord.y + p2 * patch_coord.z;
    return out;
}
