#include "mjcf_parser.h"
#include "mjcf_compiler.h"
#include <stdexcept>
#include <sstream>
#include <cstdio>

namespace joltgym {

static Vec3f ParseVec3(const char* str) {
    Vec3f v;
    if (str) sscanf(str, "%f %f %f", &v.x, &v.y, &v.z);
    return v;
}

static std::vector<float> ParseFloats(const char* str) {
    std::vector<float> result;
    if (!str) return result;
    std::istringstream iss(str);
    float val;
    while (iss >> val) result.push_back(val);
    return result;
}

MjcfModel MjcfParser::Parse(const std::string& filepath) {
    tinyxml2::XMLDocument doc;
    auto err = doc.LoadFile(filepath.c_str());
    if (err != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("Failed to load MJCF: " + filepath);
    return ParseDocument(doc);
}

MjcfModel MjcfParser::ParseString(const std::string& xml) {
    tinyxml2::XMLDocument doc;
    auto err = doc.Parse(xml.c_str(), xml.size());
    if (err != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("Failed to parse MJCF XML string");
    return ParseDocument(doc);
}

MjcfModel MjcfParser::ParseDocument(tinyxml2::XMLDocument& doc) {
    auto* root = doc.FirstChildElement("mujoco");
    if (!root)
        throw std::runtime_error("Missing <mujoco> root element");

    MjcfModel model;
    if (auto* v = root->Attribute("model")) model.name = v;

    // Parse compiler settings
    m_compiler = MjcfCompilerParser::Parse(root->FirstChildElement("compiler"));
    model.compiler = m_compiler;

    // Parse option
    model.option = MjcfCompilerParser::ParseOption(root->FirstChildElement("option"));

    // Parse defaults
    m_defaults.Parse(root->FirstChildElement("default"));

    // Parse worldbody
    auto* worldbody = root->FirstChildElement("worldbody");
    if (!worldbody)
        throw std::runtime_error("Missing <worldbody> element");

    model.worldbody.name = "world";
    // Parse direct geoms in worldbody (like floor)
    for (auto* geomElem = worldbody->FirstChildElement("geom"); geomElem;
         geomElem = geomElem->NextSiblingElement("geom")) {
        model.worldbody.geoms.push_back(ParseGeom(geomElem, ""));
    }
    // Parse child bodies
    for (auto* bodyElem = worldbody->FirstChildElement("body"); bodyElem;
         bodyElem = bodyElem->NextSiblingElement("body")) {
        model.worldbody.children.push_back(ParseBody(bodyElem, ""));
    }

    // Parse actuators
    auto* actuatorElem = root->FirstChildElement("actuator");
    if (actuatorElem) {
        model.actuators = ParseActuators(actuatorElem);
    }

    return model;
}

MjcfBody MjcfParser::ParseBody(const tinyxml2::XMLElement* bodyElem,
                                const std::string& parentClass) {
    MjcfBody body;
    if (auto* v = bodyElem->Attribute("name")) body.name = v;
    body.pos = ParseVec3(bodyElem->Attribute("pos"));

    // Parse quaternion orientation (MJCF stores as w,x,y,z — convert to our x,y,z,w)
    if (auto* v = bodyElem->Attribute("quat")) {
        float qw, qx, qy, qz;
        sscanf(v, "%f %f %f %f", &qw, &qx, &qy, &qz);
        body.quat = {qx, qy, qz, qw}; // Store as x,y,z,w (Jolt convention)
        body.has_quat = true;
    }

    // Determine active default class
    std::string activeClass = parentClass;
    if (auto* v = bodyElem->Attribute("childclass")) {
        body.childclass = v;
        activeClass = v;
    }

    // Parse geoms
    for (auto* geomElem = bodyElem->FirstChildElement("geom"); geomElem;
         geomElem = geomElem->NextSiblingElement("geom")) {
        body.geoms.push_back(ParseGeom(geomElem, activeClass));
    }

    // Parse joints
    for (auto* jointElem = bodyElem->FirstChildElement("joint"); jointElem;
         jointElem = jointElem->NextSiblingElement("joint")) {
        body.joints.push_back(ParseJoint(jointElem, activeClass));
    }

    // Recurse into child bodies
    for (auto* childElem = bodyElem->FirstChildElement("body"); childElem;
         childElem = childElem->NextSiblingElement("body")) {
        body.children.push_back(ParseBody(childElem, activeClass));
    }

    return body;
}

MjcfGeom MjcfParser::ParseGeom(const tinyxml2::XMLElement* geomElem,
                                const std::string& activeClass) {
    MjcfGeom geom;

    // Apply defaults first
    std::string cls = activeClass;
    if (auto* v = geomElem->Attribute("class")) cls = v;
    m_defaults.ApplyGeomDefaults(geom, cls);

    // Override with explicit attributes
    if (auto* v = geomElem->Attribute("name"))    geom.name = v;
    if (auto* v = geomElem->Attribute("type"))    geom.type = v;
    if (auto* v = geomElem->Attribute("pos"))     geom.pos = ParseVec3(v);
    if (auto* v = geomElem->Attribute("size"))    geom.size = ParseFloats(v);
    if (auto* v = geomElem->Attribute("condim"))  geom.condim = geomElem->IntAttribute("condim");
    if (auto* v = geomElem->Attribute("material")) geom.material = v;
    if (auto* v = geomElem->Attribute("group"))   geom.group = geomElem->IntAttribute("group");

    if (auto* v = geomElem->Attribute("fromto")) {
        std::array<float, 6> ft;
        sscanf(v, "%f %f %f %f %f %f", &ft[0], &ft[1], &ft[2], &ft[3], &ft[4], &ft[5]);
        geom.fromto = ft;
    }

    if (auto* v = geomElem->Attribute("axisangle")) {
        sscanf(v, "%f %f %f %f",
               &geom.axisangle[0], &geom.axisangle[1],
               &geom.axisangle[2], &geom.axisangle[3]);
    }

    if (auto* v = geomElem->Attribute("rgba")) {
        sscanf(v, "%f %f %f %f", &geom.rgba.x, &geom.rgba.y, &geom.rgba.z, &geom.rgba.w);
    }

    if (auto* v = geomElem->Attribute("friction")) {
        auto vals = ParseFloats(v);
        if (!vals.empty()) geom.friction = vals[0];
    }

    return geom;
}

MjcfJoint MjcfParser::ParseJoint(const tinyxml2::XMLElement* jointElem,
                                  const std::string& activeClass) {
    MjcfJoint joint;

    // Apply defaults first
    std::string cls = activeClass;
    if (auto* v = jointElem->Attribute("class")) cls = v;
    m_defaults.ApplyJointDefaults(joint, cls);

    // Override with explicit attributes
    if (auto* v = jointElem->Attribute("name"))      joint.name = v;
    if (auto* v = jointElem->Attribute("type"))      joint.type = v;
    if (auto* v = jointElem->Attribute("pos"))       joint.pos = ParseVec3(v);
    if (auto* v = jointElem->Attribute("axis"))      joint.axis = ParseVec3(v);
    if (auto* v = jointElem->Attribute("damping"))   joint.damping = std::strtof(v, nullptr);
    if (auto* v = jointElem->Attribute("stiffness")) joint.stiffness = std::strtof(v, nullptr);
    if (auto* v = jointElem->Attribute("armature"))  joint.armature = std::strtof(v, nullptr);

    if (auto* v = jointElem->Attribute("limited")) {
        joint.limited = std::string(v) == "true";
    }

    if (auto* v = jointElem->Attribute("range")) {
        sscanf(v, "%f %f", &joint.range_min, &joint.range_max);
        // Convert to radians if angle units are degrees
        if (joint.type == "hinge") {
            joint.range_min = m_compiler.ToRadians(joint.range_min);
            joint.range_max = m_compiler.ToRadians(joint.range_max);
        }
    }

    return joint;
}

std::vector<MjcfActuator> MjcfParser::ParseActuators(const tinyxml2::XMLElement* actuatorElem) {
    std::vector<MjcfActuator> actuators;

    for (auto* motorElem = actuatorElem->FirstChildElement("motor"); motorElem;
         motorElem = motorElem->NextSiblingElement("motor")) {
        MjcfActuator act;

        // Apply defaults
        m_defaults.ApplyMotorDefaults(act);

        if (auto* v = motorElem->Attribute("name"))  act.name = v;
        if (auto* v = motorElem->Attribute("joint")) act.joint = v;
        if (auto* v = motorElem->Attribute("gear"))  act.gear = std::strtof(v, nullptr);

        if (auto* v = motorElem->Attribute("ctrlrange")) {
            sscanf(v, "%f %f", &act.ctrl_min, &act.ctrl_max);
        }
        if (auto* v = motorElem->Attribute("ctrllimited")) {
            act.ctrllimited = std::string(v) == "true";
        }

        actuators.push_back(act);
    }

    return actuators;
}

} // namespace joltgym
