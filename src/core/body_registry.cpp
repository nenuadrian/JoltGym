#include "body_registry.h"
#include <stdexcept>

namespace joltgym {

void BodyRegistry::RegisterBody(const std::string& name, JPH::BodyID id) {
    m_name_to_body[name] = id;
    m_ordered_bodies.push_back(id);
    m_body_names.push_back(name);
}

void BodyRegistry::RegisterConstraint(const std::string& name, JPH::Constraint* constraint) {
    m_name_to_constraint[name] = constraint;
    m_ordered_constraints.push_back(constraint);
    m_constraint_names.push_back(name);
}

JPH::BodyID BodyRegistry::GetBody(const std::string& name) const {
    auto it = m_name_to_body.find(name);
    if (it == m_name_to_body.end())
        throw std::runtime_error("Body not found: " + name);
    return it->second;
}

JPH::Constraint* BodyRegistry::GetConstraint(const std::string& name) const {
    auto it = m_name_to_constraint.find(name);
    if (it == m_name_to_constraint.end())
        throw std::runtime_error("Constraint not found: " + name);
    return it->second;
}

void BodyRegistry::Clear() {
    m_name_to_body.clear();
    m_name_to_constraint.clear();
    m_ordered_bodies.clear();
    m_ordered_constraints.clear();
    m_body_names.clear();
    m_constraint_names.clear();
}

} // namespace joltgym
