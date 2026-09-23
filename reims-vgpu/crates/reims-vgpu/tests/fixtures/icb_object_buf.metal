#include <metal_stdlib>
using namespace metal;

// Object stage reads scale from buffer(0) — product setObjectBuffer pixel oracle.
// Object writes payload; mesh emits full-screen triangle from payload.scale.

struct Scale {
    float s;
};

struct Payload {
    float scale;
};

struct VertexOut {
    float4 position [[position]];
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, topology::triangle>;

[[object]]
void object_main(
    const device Scale &sc [[buffer(0)]],
    object_data Payload &out [[payload]],
    mesh_grid_properties mgp)
{
    out.scale = sc.s;
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
