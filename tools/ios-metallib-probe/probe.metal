// Two trivial entry points, one of each kind, so the probe has something to
// build a pipeline state from on either stage.
#include <metal_stdlib>
using namespace metal;

kernel void probe_add(device const float *a [[buffer(0)]],
                      device const float *b [[buffer(1)]],
                      device float       *o [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    o[gid] = a[gid] + b[gid];
}
