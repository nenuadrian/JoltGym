#include "collision_layers.h"

namespace joltgym {

BPLayerInterfaceImpl::BPLayerInterfaceImpl() {
    m_object_to_broadphase[Layers::STATIC]  = BroadPhaseLayers::STATIC;
    m_object_to_broadphase[Layers::DYNAMIC] = BroadPhaseLayers::DYNAMIC;
}

JPH::uint BPLayerInterfaceImpl::GetNumBroadPhaseLayers() const {
    return BroadPhaseLayers::NUM;
}

JPH::BroadPhaseLayer BPLayerInterfaceImpl::GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const {
    JPH_ASSERT(inLayer < Layers::NUM);
    return m_object_to_broadphase[inLayer];
}

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
const char* BPLayerInterfaceImpl::GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const {
    switch ((JPH::BroadPhaseLayer::Type)inLayer) {
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::STATIC:  return "STATIC";
        case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::DYNAMIC: return "DYNAMIC";
        default: return "UNKNOWN";
    }
}
#endif

bool ObjectVsBroadPhaseLayerFilterImpl::ShouldCollide(
    JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const {
    switch (inLayer1) {
        case Layers::STATIC:
            return inLayer2 == BroadPhaseLayers::DYNAMIC;
        case Layers::DYNAMIC:
            return true;
        default:
            JPH_ASSERT(false);
            return false;
    }
}

bool ObjectLayerPairFilterImpl::ShouldCollide(
    JPH::ObjectLayer inLayer1, JPH::ObjectLayer inLayer2) const {
    switch (inLayer1) {
        case Layers::STATIC:
            return inLayer2 == Layers::DYNAMIC;
        case Layers::DYNAMIC:
            return true;
        default:
            JPH_ASSERT(false);
            return false;
    }
}

JPH::ValidateResult ContactListenerImpl::OnContactValidate(
    const JPH::Body& inBody1, const JPH::Body& inBody2,
    JPH::RVec3Arg inBaseOffset,
    const JPH::CollideShapeResult& inCollisionResult) {
    return JPH::ValidateResult::AcceptAllContactsForThisBodyPair;
}

} // namespace joltgym
