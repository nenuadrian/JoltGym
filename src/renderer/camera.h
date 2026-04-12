#pragma once

#include <cmath>
#include <cstring>

namespace joltgym {

class Camera {
public:
    void SetTrackTarget(float x, float y, float z) {
        m_target[0] = x; m_target[1] = y; m_target[2] = z;
    }

    void SetDistance(float dist) { m_distance = dist; }
    void SetAzimuth(float deg) { m_azimuth = deg; }
    void SetElevation(float deg) { m_elevation = deg; }

    void GetViewMatrix(float* out) const;
    void GetProjectionMatrix(float* out, float aspect, float fov_deg = 45.0f,
                              float near_plane = 0.1f, float far_plane = 100.0f) const;
    void GetPosition(float& x, float& y, float& z) const;

    float GetAzimuth() const { return m_azimuth; }
    float GetElevation() const { return m_elevation; }
    float GetDistance() const { return m_distance; }

private:
    float m_target[3] = {0, 0, 0.5f};
    float m_distance = 3.0f;
    float m_azimuth = -90.0f;   // degrees
    float m_elevation = -20.0f; // degrees
};

} // namespace joltgym
