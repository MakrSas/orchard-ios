#include <metal_stdlib>
using namespace metal;

struct StaticSamplerFragmentIn {
    float4 position [[position]];
    float2 uv;
};

fragment float4 static_sampler_fragment(
    StaticSamplerFragmentIn in [[stage_in]],
    texture2d<float, access::sample> texture [[texture(0)]])
{
    constexpr sampler texture_sampler(
        coord::normalized,
        address::clamp_to_edge,
        filter::linear);
    return texture.sample(texture_sampler, in.uv);
}
