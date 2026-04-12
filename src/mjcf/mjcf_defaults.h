#pragma once

#include "mjcf_model.h"
#include <tinyxml2.h>
#include <unordered_map>
#include <string>

namespace joltgym {

// Stores default attribute values for each class name
struct DefaultAttributes {
    std::unordered_map<std::string, std::string> joint_attrs;
    std::unordered_map<std::string, std::string> geom_attrs;
    std::unordered_map<std::string, std::string> motor_attrs;
};

class MjcfDefaults {
public:
    void Parse(const tinyxml2::XMLElement* defaultElem);

    // Get merged attributes for a given class (or unnamed default)
    const DefaultAttributes& GetDefault(const std::string& className = "") const;
    bool HasClass(const std::string& className) const;

    // Apply defaults to a joint/geom based on class
    void ApplyJointDefaults(MjcfJoint& joint, const std::string& className) const;
    void ApplyGeomDefaults(MjcfGeom& geom, const std::string& className) const;
    void ApplyMotorDefaults(MjcfActuator& actuator) const;

private:
    void ParseDefaultElement(const tinyxml2::XMLElement* elem,
                             const std::string& parentClass,
                             DefaultAttributes inherited);

    std::unordered_map<std::string, DefaultAttributes> m_defaults;
    DefaultAttributes m_unnamed_default;
};

} // namespace joltgym
