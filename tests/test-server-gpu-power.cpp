#include "server-gpu-power.h"

#undef NDEBUG
#include <cassert>
#include <initializer_list>
#include <memory>
#include <vector>

struct fake_gpu_power_backend : server_gpu_power_backend {
    int                   init_calls     = 0;
    int                   set_calls      = 0;
    int                   shutdown_calls = 0;
    int                   fail_on_call   = 0;
    std::vector<uint32_t> applied_limits;

    bool init(int32_t device, server_gpu_power_device_info & info, std::string &) override {
        init_calls++;
        info.name                    = "fake NVIDIA device";
        info.device                  = device;
        info.original_power_limit_mw = 160000;
        info.min_power_limit_mw      = 100000;
        info.max_power_limit_mw      = 200000;
        return true;
    }

    bool set_power_limit(uint32_t power_limit_mw, std::string &) override {
        set_calls++;
        if (set_calls == fail_on_call) {
            return false;
        }
        applied_limits.push_back(power_limit_mw);
        return true;
    }

    void shutdown() override { shutdown_calls++; }
};

static server_gpu_power_phase arbitrate(std::initializer_list<server_gpu_power_slot_state> states) {
    server_gpu_power_phase_arbitrator arbitrator;
    for (const auto state : states) {
        arbitrator.observe(state);
    }
    return arbitrator.phase();
}

int main() {
    assert(arbitrate({}) == server_gpu_power_phase::idle);
    assert(arbitrate({ server_gpu_power_slot_state::idle }) == server_gpu_power_phase::idle);
    assert(arbitrate({ server_gpu_power_slot_state::wait_other }) == server_gpu_power_phase::idle);
    assert(arbitrate({ server_gpu_power_slot_state::generating }) == server_gpu_power_phase::decode);
    assert(arbitrate({ server_gpu_power_slot_state::started }) == server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::processing_prompt }) == server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::done_prompt }) == server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::generating, server_gpu_power_slot_state::processing_prompt }) ==
           server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::processing_prompt, server_gpu_power_slot_state::generating }) ==
           server_gpu_power_phase::prefill);
    assert(arbitrate({ server_gpu_power_slot_state::generating, server_gpu_power_slot_state::wait_other }) ==
           server_gpu_power_phase::decode);

    auto             backend     = std::make_unique<fake_gpu_power_backend>();
    auto *           backend_ptr = backend.get();
    server_gpu_power governor(std::move(backend));

    assert(governor.init({ 200, 165, 0 }));
    assert(governor.enabled());
    assert(backend_ptr->init_calls == 1);
    assert(backend_ptr->set_calls == 0);

    governor.update(server_gpu_power_phase::prefill);
    assert(backend_ptr->set_calls == 1);
    assert(backend_ptr->applied_limits.back() == 200000);

    governor.update(server_gpu_power_phase::prefill);
    assert(backend_ptr->set_calls == 1);

    governor.update(server_gpu_power_phase::decode);
    assert(backend_ptr->set_calls == 2);
    assert(backend_ptr->applied_limits.back() == 165000);

    governor.update(server_gpu_power_phase::idle);
    assert(backend_ptr->set_calls == 2);

    governor.update(server_gpu_power_phase::idle);
    assert(backend_ptr->set_calls == 2);

    governor.update(server_gpu_power_phase::decode);
    assert(backend_ptr->set_calls == 2);
    assert(backend_ptr->applied_limits.back() == 165000);

    governor.on_sleeping(true);
    assert(backend_ptr->set_calls == 3);
    assert(backend_ptr->applied_limits.back() == 160000);

    governor.on_sleeping(false);
    governor.update(server_gpu_power_phase::prefill);
    assert(backend_ptr->set_calls == 4);
    assert(backend_ptr->applied_limits.back() == 200000);

    governor.shutdown();
    assert(backend_ptr->set_calls == 5);
    assert(backend_ptr->applied_limits.back() == 160000);
    assert(backend_ptr->shutdown_calls == 1);

    auto             disabled_backend     = std::make_unique<fake_gpu_power_backend>();
    auto *           disabled_backend_ptr = disabled_backend.get();
    server_gpu_power disabled(std::move(disabled_backend));
    assert(disabled.init({ -1, -1, 0 }));
    assert(!disabled.enabled());
    assert(disabled_backend_ptr->init_calls == 0);
    disabled.update(server_gpu_power_phase::prefill);
    assert(disabled_backend_ptr->set_calls == 0);

    auto             invalid_backend = std::make_unique<fake_gpu_power_backend>();
    server_gpu_power invalid(std::move(invalid_backend));
    assert(!invalid.init({ 201, 165, 0 }));

    auto   failing_backend        = std::make_unique<fake_gpu_power_backend>();
    auto * failing_backend_ptr    = failing_backend.get();
    failing_backend->fail_on_call = 2;
    server_gpu_power failing(std::move(failing_backend));
    assert(failing.init({ 200, 165, 0 }));
    failing.update(server_gpu_power_phase::prefill);
    assert(failing_backend_ptr->set_calls == 1);
    failing.update(server_gpu_power_phase::decode);
    assert(!failing.enabled());
    assert(failing_backend_ptr->set_calls == 2);
    failing.update(server_gpu_power_phase::idle);
    assert(failing_backend_ptr->set_calls == 2);
    failing.shutdown();
    assert(failing_backend_ptr->set_calls == 3);
    assert(failing_backend_ptr->applied_limits.back() == 160000);

    return 0;
}
