#include "camera.h"

namespace joltgym {

static const float PI = 3.14159265358979323846f;
static float deg2rad(float d) { return d * PI / 180.0f; }

void Camera::GetPosition(float& x, float& y, float& z) const {
    float az = deg2rad(m_azimuth);
    float el = deg2rad(m_elevation);
    x = m_target[0] + m_distance * std::cos(el) * std::cos(az);
    y = m_target[1] + m_distance * std::cos(el) * std::sin(az);
    z = m_target[2] + m_distance * std::sin(-el);
}

void Camera::GetViewMatrix(float* m) const {
    float ex, ey, ez;
    GetPosition(ex, ey, ez);

    // lookAt: eye -> target, up = (0,0,1) for Z-up
    float fx = m_target[0] - ex, fy = m_target[1] - ey, fz = m_target[2] - ez;
    float fl = std::sqrt(fx*fx + fy*fy + fz*fz);
    fx /= fl; fy /= fl; fz /= fl;

    float ux = 0, uy = 0, uz = 1; // Z-up
    // right = f x up
    float rx = fy*uz - fz*uy, ry = fz*ux - fx*uz, rz = fx*uy - fy*ux;
    float rl = std::sqrt(rx*rx + ry*ry + rz*rz);
    rx /= rl; ry /= rl; rz /= rl;

    // recompute up = right x forward
    ux = ry*fz - rz*fy; uy = rz*fx - rx*fz; uz = rx*fy - ry*fx;

    // Column-major 4x4
    m[0]  = rx;  m[1]  = ux;  m[2]  = -fx; m[3]  = 0;
    m[4]  = ry;  m[5]  = uy;  m[6]  = -fy; m[7]  = 0;
    m[8]  = rz;  m[9]  = uz;  m[10] = -fz; m[11] = 0;
    m[12] = -(rx*ex + ry*ey + rz*ez);
    m[13] = -(ux*ex + uy*ey + uz*ez);
    m[14] = (fx*ex + fy*ey + fz*ez);
    m[15] = 1;
}

void Camera::GetProjectionMatrix(float* m, float aspect, float fov_deg,
                                  float near_plane, float far_plane) const {
    float fov = deg2rad(fov_deg);
    float f = 1.0f / std::tan(fov * 0.5f);

    std::memset(m, 0, sizeof(float) * 16);
    m[0]  = f / aspect;
    m[5]  = -f; // Vulkan Y is flipped
    m[10] = far_plane / (near_plane - far_plane);
    m[11] = -1.0f;
    m[14] = (near_plane * far_plane) / (near_plane - far_plane);
}

} // namespace joltgym
