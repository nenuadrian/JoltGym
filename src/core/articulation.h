/// @file articulation.h
/// @brief Articulated body — a tree of rigid bodies connected by motorized joints.
#pragma once

#include "joltgym_core.h"
#include "motor_controller.h"
#include <string>
#include <vector>
#include <memory>

namespace joltgym {

/// @brief Describes a single root degree of freedom.
///
/// Root DOFs map to the position/rotation of the root body and determine
/// the structure of `qpos` and `qvel`. HalfCheetah uses planar DOFs
/// (SlideX + SlideZ + HingeY), while Humanoid uses free DOFs
/// (FreeX/Y/Z + QuatW/X/Y/Z).
struct RootDOF {
    /// @brief Type of root degree of freedom.
    enum class Type {
        SlideX, SlideZ, HingeY,    ///< 2D planar (HalfCheetah)
        FreeX, FreeY, FreeZ,       ///< 3D position (Humanoid)
        QuatW, QuatX, QuatY, QuatZ ///< 3D orientation quaternion (Humanoid)
    };
    std::string name;  ///< Joint name from the MJCF model.
    Type type;         ///< DOF type.

    /// @brief Whether this DOF contributes to qpos.
    bool IsPositionDOF() const {
        return type != Type::QuatW;
    }

    /// @brief Number of qpos entries this DOF type adds (always 1).
    static int QPosDim(Type t) { return 1; }

    /// @brief Whether this DOF is part of a quaternion group.
    ///
    /// Quaternion DOFs have 4 qpos entries but map to only 3 qvel entries
    /// (angular velocity).
    static bool IsQuatComponent(Type t) {
        return t == Type::QuatW || t == Type::QuatX ||
               t == Type::QuatY || t == Type::QuatZ;
    }
};

/// @brief An articulated body composed of rigid bodies connected by motorized joints.
///
/// Manages the kinematic tree, motor controllers, and state extraction
/// (qpos/qvel) for a single robot. Supports both 2D planar roots
/// (HalfCheetah) and 3D free roots (Humanoid).
class Articulation {
public:
    /// @param name Articulation name (e.g. "halfcheetah", "humanoid").
    Articulation(const std::string& name);

    /// @brief Set the root body of the kinematic tree.
    void SetRootBody(JPH::BodyID root_body);

    /// @brief Register a body in the kinematic tree.
    void AddBody(const std::string& name, JPH::BodyID id);

    /// @brief Add a motor controller for an actuated joint.
    void AddMotor(std::unique_ptr<MotorController> motor);

    /// @brief Add a root degree of freedom.
    void AddRootDOF(const RootDOF& dof);

    /// @name State dimensions
    /// @{
    int GetQPosDim() const;   ///< Number of generalized position coordinates.
    int GetQVelDim() const;   ///< Number of generalized velocity coordinates.
    int GetActionDim() const; ///< Number of actuated joints.
    /// @}

    /// @name State extraction
    /// @{
    /// @brief Write generalized positions to @p out.
    void GetQPos(float* out, const JPH::BodyInterface& body_interface) const;
    /// @brief Write generalized velocities to @p out.
    void GetQVel(float* out, const JPH::BodyInterface& body_interface) const;
    /// @}

    /// @brief Apply normalized actions to all motor controllers.
    /// @param actions Array of normalized actions in [-1, 1].
    /// @param count   Number of actions (must equal GetActionDim()).
    void ApplyActions(const float* actions, int count);

    /// @brief Apply passive damping and stiffness torques.
    void ApplyPassiveForces(JPH::BodyInterface& body_interface);

    /// @name Accessors
    /// @{
    const std::string& GetName() const { return m_name; }
    JPH::BodyID GetRootBody() const { return m_root_body; }
    const std::vector<JPH::BodyID>& GetBodies() const { return m_bodies; }
    const std::vector<std::string>& GetBodyNames() const { return m_body_names; }
    const std::vector<std::unique_ptr<MotorController>>& GetMotors() const { return m_motors; }
    MotorController* GetMotor(size_t index) { return m_motors[index].get(); }
    /// @}

    /// @name Root body queries
    /// @{
    float GetRootX(const JPH::BodyInterface& body_interface) const;          ///< Root X position.
    float GetRootXVelocity(const JPH::BodyInterface& body_interface) const;  ///< Root X velocity.
    float GetRootZ(const JPH::BodyInterface& body_interface) const;          ///< Root Z position (height).
    /// @}

    /// @brief Whether this articulation has a free (6DOF) root joint.
    bool HasFreeRoot() const { return m_has_free_root; }
    void SetHasFreeRoot(bool v) { m_has_free_root = v; }

private:
    bool m_has_free_root = false;
    std::string m_name;
    JPH::BodyID m_root_body;
    std::vector<std::string> m_body_names;
    std::vector<JPH::BodyID> m_bodies;
    std::vector<std::unique_ptr<MotorController>> m_motors;
    std::vector<RootDOF> m_root_dofs;
};

} // namespace joltgym
