#include <metal_stdlib>
using namespace metal;

// Mesh-only full-screen triangle (oversized) for ICB drawMeshThreads oracle.
// Single export per metallib — product fill requires exactly one function name.

struct VertexOut {
    float4 position [[position]];
};

using MeshOut = metal::mesh<VertexOut, void, 3, 1, topology::triangle>;

[[mesh]]
void mesh_main(uint tid [[thread_index_in_threadgroup]], MeshOut out) {
    if (tid == 0) {
        out.set_vertex(0, VertexOut{float4(-1.0f, -1.0f, 0.0f, 1.0f)});
        out.set_vertex(1, VertexOut{float4(3.0f, -1.0f, 0.0f, 1.0f)});
        out.set_vertex(2, VertexOut{float4(-1.0f, 3.0f, 0.0f, 1.0f)});
        out.set_index(0, 0);
        out.set_index(1, 1);
        out.set_index(2, 2);
        out.set_primitive_count(1);
    }
}
