#pragma once

#include "joltgym_core.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace joltgym {

class BodyRegistry {
public:
    void RegisterBody(const std::string& name, JPH::BodyID id);
    void RegisterConstraint(const std::string& name, JPH::Constraint* constraint);

    JPH::BodyID GetBody(const std::string& name) const;
    JPH::Constraint* GetConstraint(const std::string& name) const;

    const std::vector<JPH::BodyID>& GetOrderedBodies() const { return m_ordered_bodies; }
    const std::vector<JPH::Constraint*>& GetOrderedConstraints() const { return m_ordered_constraints; }
    const std::vector<std::string>& GetBodyNames() const { return m_body_names; }
    const std::vector<std::string>& GetConstraintNames() const { return m_constraint_names; }

    size_t NumBodies() const { return m_ordered_bodies.size(); }
    size_t NumConstraints() const { return m_ordered_constraints.size(); }

    void Clear();

private:
    std::unordered_map<std::string, JPH::BodyID> m_name_to_body;
    std::unordered_map<std::string, JPH::Constraint*> m_name_to_constraint;
    std::vector<JPH::BodyID> m_ordered_bodies;
    std::vector<JPH::Constraint*> m_ordered_constraints;
    std::vector<std::string> m_body_names;
    std::vector<std::string> m_constraint_names;
};

} // namespace joltgym
