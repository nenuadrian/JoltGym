#include "joltgym_core.h"

#include <cstdarg>
#include <cstdio>
#include <thread>

// Jolt requires a custom trace and assert implementation
static void JoltTraceImpl(const char* inFMT, ...) {
    va_list list;
    va_start(list, inFMT);
    vfprintf(stderr, inFMT, list);
    fprintf(stderr, "\n");
    va_end(list);
}

#ifdef JPH_ENABLE_ASSERTS
static bool JoltAssertFailedImpl(const char* inExpression, const char* inMessage,
                                  const char* inFile, JPH::uint inLine) {
    fprintf(stderr, "JOLT ASSERT: %s:%u: (%s) %s\n",
            inFile, inLine, inExpression, inMessage ? inMessage : "");
    return true; // break into debugger
}
#endif

namespace joltgym {

bool JoltGymCore::s_initialized = false;
std::unique_ptr<JPH::JobSystemThreadPool> JoltGymCore::s_job_system;
std::once_flag JoltGymCore::s_init_flag;

void JoltGymCore::Init() {
    std::call_once(s_init_flag, []() {
        // Must register allocator FIRST before any Jolt allocations
        JPH::RegisterDefaultAllocator();

        JPH::Trace = JoltTraceImpl;
        #ifdef JPH_ENABLE_ASSERTS
        JPH::AssertFailed = JoltAssertFailedImpl;
        #endif

        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();

        int num_threads = std::max(1, (int)std::thread::hardware_concurrency() - 1);
        s_job_system = std::make_unique<JPH::JobSystemThreadPool>(
            JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, num_threads);

        s_initialized = true;
    });
}

void JoltGymCore::Shutdown() {
    if (s_initialized) {
        s_job_system.reset();
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        s_initialized = false;
    }
}

bool JoltGymCore::IsInitialized() {
    return s_initialized;
}

JPH::JobSystemThreadPool& JoltGymCore::GetJobSystem() {
    if (!s_initialized) Init();
    return *s_job_system;
}

} // namespace joltgym
