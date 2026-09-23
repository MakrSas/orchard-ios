// The vertex half of the pipeline-assembly probe.
//
// It exists to be a stage, not to draw: one `vec4` attribute at location 0
// passed straight through, which is the vertex input the probe's key declares.
// Anything more would make a failed pipeline creation ambiguous between the
// assembly under test and the shader.
#version 450

layout(location = 0) in vec4 position;

void main() {
    gl_Position = position;
}
