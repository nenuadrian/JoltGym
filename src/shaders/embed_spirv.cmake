# embed_spirv.cmake - Converts a SPIR-V binary to a C header with an embedded array
# Usage: cmake -P embed_spirv.cmake <input.spv> <output.h> <variable_name>

set(INPUT_FILE ${CMAKE_ARGV3})
set(OUTPUT_FILE ${CMAKE_ARGV4})
set(VAR_NAME ${CMAKE_ARGV5})

file(READ ${INPUT_FILE} SPIRV_DATA HEX)
string(LENGTH "${SPIRV_DATA}" SPIRV_HEX_LEN)
math(EXPR SPIRV_BYTE_COUNT "${SPIRV_HEX_LEN} / 2")

# Convert hex string to comma-separated byte array
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," SPIRV_BYTES "${SPIRV_DATA}")
# Add newlines every 16 bytes for readability
string(REGEX REPLACE "(0x[0-9a-f][0-9a-f],){16}" "\\0\n    " SPIRV_BYTES "${SPIRV_BYTES}")

file(WRITE ${OUTPUT_FILE}
"#pragma once
#include <cstdint>
#include <cstddef>

static const uint8_t ${VAR_NAME}_data[] = {
    ${SPIRV_BYTES}
};

static const size_t ${VAR_NAME}_size = ${SPIRV_BYTE_COUNT};
")
