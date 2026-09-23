#include <metal_stdlib>
using namespace metal;

// Mesh stage reads scale from buffer(0) — product setMeshBuffer pixel oracle.
// scale == 1.0 → full-screen triangle; wrong/missing bind → black / fail.

struct Scale {
    float s;
};

struct VertexOut {
    float4 position [[position]];
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, topology::triangle>;

[[mesh]]
void mesh_main(
    const device Scale &sc [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    MeshOut out)
{
    if (tid == 0) {
        float s = sc.s;
        out.set_vertex(0, VertexOut{float4(-1.0f * s, -1.0f * s, 0.0f, 1.0f)});
        out.set_vertex(1, VertexOut{float4(3.0f * s, -1.0f * s, 0.0f, 1.0f)});
        out.set_vertex(2, VertexOut{float4(-1.0f * s, 3.0f * s, 0.0f, 1.0f)});
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
        out.set_primitive_count(1);
    }
}
