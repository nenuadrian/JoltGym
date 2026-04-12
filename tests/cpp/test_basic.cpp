#include <cstdio>
#include <memory>
#include "core/joltgym_core.h"
#include "core/physics_world.h"
#include "mjcf/mjcf_parser.h"
#include "mjcf/mjcf_to_jolt.h"
#include "core/state_extractor.h"

int main() {
    fprintf(stderr, "test: parsing MJCF...\n"); fflush(stderr);

    joltgym::MjcfParser parser;
    auto model = parser.Parse("python/joltgym/assets/half_cheetah.xml");
    fprintf(stderr, "test: parsed model '%s' with %zu actuators\n",
            model.name.c_str(), model.actuators.size()); fflush(stderr);

    fprintf(stderr, "test: init physics world...\n"); fflush(stderr);
    auto world_ptr = std::make_unique<joltgym::PhysicsWorld>();
    world_ptr->Init();
    fprintf(stderr, "test: world initialized\n"); fflush(stderr);

    fprintf(stderr, "test: building from model...\n"); fflush(stderr);
    joltgym::MjcfToJolt builder;
    auto* artic = builder.Build(model, *world_ptr);
    fprintf(stderr, "test: articulation created: %s\n",
            artic ? artic->GetName().c_str() : "null"); fflush(stderr);

    if (artic) {
        fprintf(stderr, "test: qpos_dim=%d qvel_dim=%d action_dim=%d\n",
                artic->GetQPosDim(), artic->GetQVelDim(), artic->GetActionDim()); fflush(stderr);

        joltgym::StateExtractor state(artic, world_ptr.get());
        fprintf(stderr, "test: obs_dim=%d\n", state.GetObsDim()); fflush(stderr);

        for (int i = 0; i < 100; i++) {
            float actions[] = {0, 0, 0, 0, 0, 0};
            artic->ApplyActions(actions, 6);
            world_ptr->Step(0.01f, 1);
        }
        fprintf(stderr, "test: 100 steps completed!\n"); fflush(stderr);
        fprintf(stderr, "test: root_x = %.4f\n", state.GetRootX()); fflush(stderr);
    }

    fprintf(stderr, "test: SUCCESS!\n");
    return 0;
}
