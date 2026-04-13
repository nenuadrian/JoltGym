/// @file mjcf_model.h
/// @brief Data structures representing a parsed MuJoCo XML (MJCF) model.
#pragma once

#include <string>
#include <vector>
#include <array>
#include <optional>
#include <cmath>

namespace joltgym {

/// @brief 3D float vector with basic operations.
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

/// @brief 4D float vector (quaternion or RGBA color).
struct Vec4f {
    float x = 0, y = 0, z = 0, w = 1;
};

/// @brief A collision/visual geometry element parsed from `<geom>`.
///
/// Supported types: capsule, sphere, box, cylinder, plane.
struct MjcfGeom {
    std::string name;                                      ///< Geom name.
    std::string type = "capsule";                          ///< Geometry type.
    Vec3f pos;                                             ///< Local position.
    std::array<float, 4> axisangle = {0, 0, 1, 0};        ///< Axis-angle rotation.
    std::optional<std::array<float, 6>> fromto;            ///< Alternative `fromto` specification.
    std::vector<float> size;                               ///< Size parameters (varies by type).
    Vec4f rgba = {0.8f, 0.6f, 0.4f, 1.0f};                ///< Color.
    int condim = 3;                                        ///< Contact dimensionality.
    float friction = 0.4f;                                 ///< Friction coefficient.
    std::string material;                                  ///< Material name.
    std::string geom_class;                                ///< Default class name.
    int group = 0;                                         ///< Geom group.
};

/// @brief A joint element parsed from `<joint>`.
///
/// Supported types: hinge (revolute), slide (prismatic), ball, free.
struct MjcfJoint {
    std::string name;                     ///< Joint name.
    std::string type = "hinge";           ///< Joint type.
    Vec3f pos;                            ///< Anchor position.
    Vec3f axis = {0, 0, 1};              ///< Rotation/translation axis.
    float range_min = 0;                  ///< Lower joint limit.
    float range_max = 0;                  ///< Upper joint limit.
    bool limited = true;                  ///< Whether joint limits are enforced.
    float damping = 0.0f;                ///< Passive damping coefficient.
    float stiffness = 0.0f;              ///< Passive stiffness coefficient.
    float armature = 0.0f;               ///< Rotor inertia.
    std::string joint_class;              ///< Default class name.
};

/// @brief A body element parsed from `<body>`, forming a tree hierarchy.
struct MjcfBody {
    std::string name;                      ///< Body name.
    Vec3f pos;                             ///< Position relative to parent.
    Vec4f quat = {0, 0, 0, 1};            ///< Orientation (x, y, z, w).
    bool has_quat = false;                 ///< Whether a quaternion was explicitly set.
    std::string childclass;                ///< Default class for children.
    std::vector<MjcfGeom> geoms;           ///< Collision/visual geometries.
    std::vector<MjcfJoint> joints;         ///< Joint definitions.
    std::vector<MjcfBody> children;        ///< Child bodies.
};

/// @brief An actuator element parsed from `<actuator><motor>`.
struct MjcfActuator {
    std::string name;                      ///< Actuator name.
    std::string joint;                     ///< Name of the joint this actuator drives.
    float gear = 1.0f;                     ///< Gear ratio (torque multiplier).
    float ctrl_min = -1.0f;               ///< Minimum control input.
    float ctrl_max = 1.0f;                ///< Maximum control input.
    bool ctrllimited = true;               ///< Whether control limits are enforced.
};

/// @brief Compiler directives parsed from `<compiler>`.
struct MjcfCompiler {
    std::string angle = "degree";          ///< Angle units: "degree" or "radian".
    std::string coordinate = "global";     ///< Coordinate frame: "global" or "local".
    bool inertiafromgeom = false;           ///< Compute inertia from geom shapes.
    float settotalmass = -1.0f;            ///< Override total mass (-1 = unset).

    /// @brief Convert a value to radians based on the configured angle unit.
    float ToRadians(float val) const {
        if (angle == "degree") return val * (float)M_PI / 180.0f;
        return val;
    }
};

/// @brief Simulation options parsed from `<option>`.
struct MjcfOption {
    Vec3f gravity = {0, 0, -9.81f};        ///< Gravity vector (m/s^2).
    float timestep = 0.002f;               ///< Default simulation timestep (seconds).
};

/// @brief Top-level parsed representation of a MuJoCo XML model.
struct MjcfModel {
    std::string name;                      ///< Model name.
    MjcfCompiler compiler;                 ///< Compiler directives.
    MjcfOption option;                     ///< Simulation options.
    MjcfBody worldbody;                    ///< Root of the body tree.
    std::vector<MjcfActuator> actuators;   ///< Motor definitions.
};

} // namespace joltgym
