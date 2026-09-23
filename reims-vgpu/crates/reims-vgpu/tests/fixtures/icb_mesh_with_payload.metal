#include <metal_stdlib>
using namespace metal;

// Mesh-stage metallib that consumes object-stage Payload (mesh SPI
// mesh_func_ref tag 0x02 under section tag 0x14). Payload layout must match
// icb_object_stage.metal.

struct VertexOut {
    float4 position [[position]];
};

struct Payload {
    float scale;
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, topology::triangle>;

[[mesh]]
void mesh_main(
    object_data Payload const& in [[payload]],
    uint tid [[thread_index_in_threadgroup]],
    MeshOut out)
{
    if (tid == 0) {
        float s = in.scale;
        out.set_vertex(0, VertexOut{float4(-1.0f * s, -1.0f * s, 0.0f, 1.0f)});
        out.set_vertex(1, VertexOut{float4(3.0f * s, -1.0f * s, 0.0f, 1.0f)});
        out.set_vertex(2, VertexOut{float4(-1.0f * s, 3.0f * s, 0.0f, 1.0f)});
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
        out.set_primitive_count(1);
    }
}
