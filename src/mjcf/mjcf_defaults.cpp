#include "mjcf_defaults.h"
#include <cstdlib>
#include <cstring>

namespace joltgym {

static std::unordered_map<std::string, std::string> ParseAttrs(const tinyxml2::XMLElement* elem) {
    std::unordered_map<std::string, std::string> attrs;
    if (!elem) return attrs;
    for (auto* attr = elem->FirstAttribute(); attr; attr = attr->Next()) {
        attrs[attr->Name()] = attr->Value();
    }
    return attrs;
}

static void MergeAttrs(std::unordered_map<std::string, std::string>& base,
                       const std::unordered_map<std::string, std::string>& override_attrs) {
    for (auto& [k, v] : override_attrs) {
        base[k] = v;
    }
}

void MjcfDefaults::ParseDefaultElement(const tinyxml2::XMLElement* elem,
                                        const std::string& parentClass,
                                        DefaultAttributes inherited) {
    const char* className = elem->Attribute("class");
    std::string cls = className ? className : parentClass;

    // Parse element-specific defaults
    if (auto* jointElem = elem->FirstChildElement("joint")) {
        auto attrs = ParseAttrs(jointElem);
        MergeAttrs(inherited.joint_attrs, attrs);
    }
    if (auto* geomElem = elem->FirstChildElement("geom")) {
        auto attrs = ParseAttrs(geomElem);
        MergeAttrs(inherited.geom_attrs, attrs);
    }
    if (auto* motorElem = elem->FirstChildElement("motor")) {
        auto attrs = ParseAttrs(motorElem);
        MergeAttrs(inherited.motor_attrs, attrs);
    }

    // Store for this class
    if (!cls.empty()) {
        m_defaults[cls] = inherited;
    } else {
        m_unnamed_default = inherited;
    }

    // Recurse into nested defaults
    for (auto* child = elem->FirstChildElement("default"); child;
         child = child->NextSiblingElement("default")) {
        ParseDefaultElement(child, cls, inherited);
    }
}

void MjcfDefaults::Parse(const tinyxml2::XMLElement* defaultElem) {
    if (!defaultElem) return;
    DefaultAttributes base;
    ParseDefaultElement(defaultElem, "", base);
}

const DefaultAttributes& MjcfDefaults::GetDefault(const std::string& className) const {
    if (!className.empty()) {
        auto it = m_defaults.find(className);
        if (it != m_defaults.end()) return it->second;
    }
    return m_unnamed_default;
}

bool MjcfDefaults::HasClass(const std::string& className) const {
    return m_defaults.find(className) != m_defaults.end();
}

static float GetFloatAttr(const std::unordered_map<std::string, std::string>& attrs,
                          const std::string& name, float fallback) {
    auto it = attrs.find(name);
    if (it != attrs.end()) return std::strtof(it->second.c_str(), nullptr);
    return fallback;
}

static std::string GetStrAttr(const std::unordered_map<std::string, std::string>& attrs,
                               const std::string& name, const std::string& fallback) {
    auto it = attrs.find(name);
    if (it != attrs.end()) return it->second;
    return fallback;
}

static bool GetBoolAttr(const std::unordered_map<std::string, std::string>& attrs,
                         const std::string& name, bool fallback) {
    auto it = attrs.find(name);
    if (it != attrs.end()) return it->second == "true";
    return fallback;
}

void MjcfDefaults::ApplyJointDefaults(MjcfJoint& joint, const std::string& className) const {
    const auto& def = GetDefault(className);
    auto& attrs = def.joint_attrs;

    joint.armature = GetFloatAttr(attrs, "armature", joint.armature);
    joint.damping = GetFloatAttr(attrs, "damping", joint.damping);
    joint.stiffness = GetFloatAttr(attrs, "stiffness", joint.stiffness);
    joint.limited = GetBoolAttr(attrs, "limited", joint.limited);
    joint.type = GetStrAttr(attrs, "type", joint.type);
}

void MjcfDefaults::ApplyGeomDefaults(MjcfGeom& geom, const std::string& className) const {
    const auto& def = GetDefault(className);
    auto& attrs = def.geom_attrs;

    geom.type = GetStrAttr(attrs, "type", geom.type);
    geom.friction = GetFloatAttr(attrs, "friction", geom.friction);
    geom.condim = (int)GetFloatAttr(attrs, "condim", (float)geom.condim);

    auto rgba_it = attrs.find("rgba");
    if (rgba_it != attrs.end()) {
        sscanf(rgba_it->second.c_str(), "%f %f %f %f",
               &geom.rgba.x, &geom.rgba.y, &geom.rgba.z, &geom.rgba.w);
    }
}

void MjcfDefaults::ApplyMotorDefaults(MjcfActuator& actuator) const {
    auto& attrs = m_unnamed_default.motor_attrs;

    auto ctrlrange_it = attrs.find("ctrlrange");
    if (ctrlrange_it != attrs.end()) {
        sscanf(ctrlrange_it->second.c_str(), "%f %f",
               &actuator.ctrl_min, &actuator.ctrl_max);
    }
    actuator.ctrllimited = GetBoolAttr(attrs, "ctrllimited", actuator.ctrllimited);
}

} // namespace joltgym
