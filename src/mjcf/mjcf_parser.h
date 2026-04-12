#pragma once

#include "mjcf_model.h"
#include "mjcf_defaults.h"
#include <string>

namespace joltgym {

class MjcfParser {
public:
    MjcfModel Parse(const std::string& filepath);
    MjcfModel ParseString(const std::string& xml);

    const MjcfDefaults& GetDefaults() const { return m_defaults; }

private:
    MjcfModel ParseDocument(tinyxml2::XMLDocument& doc);
    MjcfBody ParseBody(const tinyxml2::XMLElement* bodyElem,
                       const std::string& parentClass);
    MjcfGeom ParseGeom(const tinyxml2::XMLElement* geomElem,
                       const std::string& activeClass);
    MjcfJoint ParseJoint(const tinyxml2::XMLElement* jointElem,
                         const std::string& activeClass);
    std::vector<MjcfActuator> ParseActuators(const tinyxml2::XMLElement* actuatorElem);

    MjcfDefaults m_defaults;
    MjcfCompiler m_compiler;
};

} // namespace joltgym
