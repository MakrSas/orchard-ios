// The fragment half of the pipeline-assembly probe. One colour attachment,
// which is what the probe's compatibility declares.
#version 450

layout(location = 0) out vec4 colour;

void main() {
    colour = vec4(1.0, 0.0, 0.0, 1.0);
}
