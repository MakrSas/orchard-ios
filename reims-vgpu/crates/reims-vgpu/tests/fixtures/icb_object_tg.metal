#include <metal_stdlib>
using namespace metal;

// Object stage uses threadgroup memory at index 0 (setObjectThreadgroupMemoryLength).
// Writes scale=1.0 into TG, then into payload. Mesh draws full-screen triangle.
// Without a correct non-zero TG length (multiple of 16), this path is invalid.

struct Payload {
    float scale;
};

struct VertexOut {
    float4 position [[position]];
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, topology::triangle>;

[[object]]
void object_main(
    object_data Payload &out [[payload]],
    mesh_grid_properties mgp,
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float *tg [[threadgroup(0)]])
{
    if (tid == 0) {
        tg[0] = 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out.scale = tg[0];
    mgp.set_threadgroups_per_grid(uint3(1, 1, 1));
}

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
