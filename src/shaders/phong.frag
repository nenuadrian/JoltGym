#version 450

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec3 light_dir;
    float ambient;
    vec3 light_color;
    float _pad;
    vec3 camera_pos;
} scene;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 L = normalize(-scene.light_dir);
    vec3 V = normalize(scene.camera_pos - fragWorldPos);
    vec3 H = normalize(L + V);

    // Blinn-Phong
    float diffuse = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 32.0);

    vec3 color = fragColor.rgb * (scene.ambient + diffuse * scene.light_color)
               + specular * scene.light_color * 0.3;

    // Ground plane checker pattern
    if (fragColor.a < 0.5) {
        float checker = mod(floor(fragWorldPos.x) + floor(fragWorldPos.y), 2.0);
        color = mix(vec3(0.15, 0.25, 0.35), vec3(0.25, 0.35, 0.45), checker);
        color *= (scene.ambient + diffuse * scene.light_color);
    }

    outColor = vec4(color, 1.0);
}
