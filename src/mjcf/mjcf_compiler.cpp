#include "mjcf_compiler.h"
#include <cstdlib>
#include <cstdio>

namespace joltgym {

MjcfCompiler MjcfCompilerParser::Parse(const tinyxml2::XMLElement* elem) {
    MjcfCompiler compiler;
    if (!elem) return compiler;

    if (auto* v = elem->Attribute("angle"))          compiler.angle = v;
    if (auto* v = elem->Attribute("coordinate"))     compiler.coordinate = v;
    if (auto* v = elem->Attribute("inertiafromgeom")) compiler.inertiafromgeom = std::string(v) == "true";
    if (auto* v = elem->Attribute("settotalmass"))   compiler.settotalmass = std::strtof(v, nullptr);

    return compiler;
}

MjcfOption MjcfCompilerParser::ParseOption(const tinyxml2::XMLElement* elem) {
    MjcfOption option;
    if (!elem) return option;

    if (auto* v = elem->Attribute("gravity")) {
        sscanf(v, "%f %f %f", &option.gravity.x, &option.gravity.y, &option.gravity.z);
    }
    if (auto* v = elem->Attribute("timestep")) {
        option.timestep = std::strtof(v, nullptr);
    }

    return option;
}

} // namespace joltgym
