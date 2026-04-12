#pragma once

#include <string>
#include <vector>
#include <array>
#include <optional>
#include <cmath>

namespace joltgym {

struct Vec3f {
    float x = 0, y = 0, z = 0;
    Vec3f() = default;
    Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3f normalized() const { float l = length(); return {x/l, y/l, z/l}; }
    Vec3f operator+(const Vec3f& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3f operator-(const Vec3f& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3f operator*(float s) const { return {x*s, y*s, z*s}; }
    float dot(const Vec3f& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3f cross(const Vec3f& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
};

struct Vec4f {
    float x = 0, y = 0, z = 0, w = 1;
};

struct MjcfGeom {
    std::string name;
    std::string type = "capsule"; // capsule, sphere, box, cylinder, plane
    Vec3f pos;
    std::array<float, 4> axisangle = {0, 0, 1, 0}; // axis-angle rotation
    std::optional<std::array<float, 6>> fromto; // fromto="x1 y1 z1 x2 y2 z2"
    std::vector<float> size; // varies by type
    Vec4f rgba = {0.8f, 0.6f, 0.4f, 1.0f};
    int condim = 3;
    float friction = 0.4f;
    std::string material;
    std::string geom_class; // default class
    int group = 0;
};

struct MjcfJoint {
    std::string name;
    std::string type = "hinge"; // hinge, slide, ball, free
    Vec3f pos;
    Vec3f axis = {0, 0, 1};
    float range_min = 0;
    float range_max = 0;
    bool limited = true;
    float damping = 0.01f;
    float stiffness = 8.0f;
    float armature = 0.1f;
    std::string joint_class;
};

struct MjcfBody {
    std::string name;
    Vec3f pos;
    std::string childclass;
    std::vector<MjcfGeom> geoms;
    std::vector<MjcfJoint> joints;
    std::vector<MjcfBody> children;
};

struct MjcfActuator {
    std::string name;
    std::string joint; // joint name this actuator acts on
    float gear = 1.0f;
    float ctrl_min = -1.0f;
    float ctrl_max = 1.0f;
    bool ctrllimited = true;
};

struct MjcfCompiler {
    std::string angle = "degree"; // "degree" or "radian"
    std::string coordinate = "global"; // "global" or "local"
    bool inertiafromgeom = false;
    float settotalmass = -1.0f; // -1 means unset

    float ToRadians(float val) const {
        if (angle == "degree") return val * (float)M_PI / 180.0f;
        return val;
    }
};

struct MjcfOption {
    Vec3f gravity = {0, 0, -9.81f};
    float timestep = 0.002f;
};

struct MjcfModel {
    std::string name;
    MjcfCompiler compiler;
    MjcfOption option;
    MjcfBody worldbody; // root of body tree
    std::vector<MjcfActuator> actuators;
};

} // namespace joltgym
