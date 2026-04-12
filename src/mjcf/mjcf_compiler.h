#pragma once

#include "mjcf_model.h"
#include <tinyxml2.h>

namespace joltgym {

class MjcfCompilerParser {
public:
    static MjcfCompiler Parse(const tinyxml2::XMLElement* compilerElem);
    static MjcfOption ParseOption(const tinyxml2::XMLElement* optionElem);
};

} // namespace joltgym
