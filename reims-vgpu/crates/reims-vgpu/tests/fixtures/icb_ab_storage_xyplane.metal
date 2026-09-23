#include <metal_stdlib>
using namespace metal;
struct StorageArgs {
    texture2d<uint, access::write> out [[id(0)]];
};
kernel void icb_ab_storage_xyplane(
    constant StorageArgs &args [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    args.out.write(uint4(gid.x, gid.y, 5u, 255u), gid);
}
