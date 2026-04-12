#version 450

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec3 light_dir;
    float ambient;
    vec3 light_color;
    float _pad;
    vec3 camera_pos;
} scene;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec4 fragColor;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    fragNormal = mat3(transpose(inverse(push.model))) * inNormal;
    fragColor = push.color;
    gl_Position = scene.proj * scene.view * worldPos;
}
