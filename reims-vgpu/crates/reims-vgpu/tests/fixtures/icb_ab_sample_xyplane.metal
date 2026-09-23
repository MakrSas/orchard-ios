#include <metal_stdlib>
using namespace metal;
struct SampleArgs {
    texture2d<float, access::sample> in [[id(0)]];
    texture2d<uint, access::write> out [[id(1)]];
    sampler s [[id(2)]];
};
kernel void icb_ab_sample_xyplane(
    constant SampleArgs &args [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    float2 uv = (float2(gid) + 0.5) / float2(args.out.get_width(), args.out.get_height());
    float4 c = args.in.sample(args.s, uv);
    args.out.write(uint4(uint(c.r * 255.0 + 0.5),
                         uint(c.g * 255.0 + 0.5),
                         uint(c.b * 255.0 + 0.5),
                         uint(c.a * 255.0 + 0.5)), gid);
}
