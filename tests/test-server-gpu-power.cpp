#include "server-gpu-power.h"

#undef NDEBUG
#include <cassert>
#include <initializer_list>
#include <memory>
#include <vector>

struct fake_gpu_power_backend : server_gpu_power_backend {
    int                   init_calls         = 0;
    int                   set_calls          = 0;
    int                   set_mem_calls      = 0;
    int                   reset_mem_calls    = 0;
    int                   set_offset_calls   = 0;
    int                   reset_offset_calls = 0;
    int                   shutdown_calls     = 0;
    int                   fail_on_call       = 0;
    std::vector<uint32_t> applied_limits;
    std::vector<uint32_t> applied_mem_clocks;
    std::vector<int32_t>  applied_offsets;

    bool init(int32_t device, server_gpu_power_device_info & info, std::string &) override {
        init_calls++;
        info.name                     = "fake NVIDIA device";
        info.device                   = device;
        info.original_power_limit_mw  = 160000;
        info.min_power_limit_mw       = 100000;
        info.max_power_limit_mw       = 200000;
        info.supported_mem_clocks_mhz = { 10501, 10251, 5001, 810, 405 };
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

    bool set_memory_locked_clocks(uint32_t, uint32_t max_mhz, std::string &) override {
        set_mem_calls++;
        if (set_mem_calls == fail_on_call) {
            return false;
        }
        applied_mem_clocks.push_back(max_mhz);
        return true;
    }

    bool reset_memory_locked_clocks(std::string &) override {
        reset_mem_calls++;
        return true;
    }

    bool set_memory_clock_offset(int32_t offset_mhz, std::string &) override {
        set_offset_calls++;
        applied_offsets.push_back(offset_mhz);
        return true;
    }

    bool reset_memory_clock_offset(std::string &) override {
        reset_offset_calls++;
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

    // 1. Test pure power governor
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        assert(governor.init({ 200, 165, -1, -1, 0 }));
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
    }

    // 2. Test pure memory governor (decode only, discrete stock)
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        assert(governor.init({ -1, -1, 10501, -1, 0 }));
        assert(governor.enabled());
        assert(backend_ptr->init_calls == 1);
        assert(backend_ptr->set_mem_calls == 0);

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_mem_calls == 0);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);

        // Deduplicated
        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);

        // Transition to idle resets memory clock
        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->reset_mem_calls == 1);

        // Transition back to decode relocks memory
        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 2);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);

        governor.shutdown();
        assert(backend_ptr->reset_mem_calls == 2);
    }

    // 3. Test combined power + memory governor
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        assert(governor.init({ 200, 165, 10501, 10251, 0 }));
        assert(governor.enabled());

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_calls == 1);
        assert(backend_ptr->applied_limits.back() == 200000);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10251);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_calls == 2);
        assert(backend_ptr->applied_limits.back() == 165000);
        assert(backend_ptr->set_mem_calls == 2);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);

        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->reset_mem_calls == 1);

        governor.shutdown();
        assert(backend_ptr->set_calls == 3);
        assert(backend_ptr->applied_limits.back() == 160000);
    }

    // 4. Test unsupported memory clock rejected (e.g. non-overclock value not in supported list)
    {
        auto             backend = std::make_unique<fake_gpu_power_backend>();
        server_gpu_power governor(std::move(backend));
        // 7000 is between 5001 and 10251, not in discrete list and not an overclock > 10501
        assert(!governor.init({ -1, -1, 7000, -1, 0 }));
    }

    // 5. Test disabled backend
    {
        auto             disabled_backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           disabled_backend_ptr = disabled_backend.get();
        server_gpu_power disabled(std::move(disabled_backend));
        assert(disabled.init({ -1, -1, -1, -1, 0 }));
        assert(!disabled.enabled());
        assert(disabled_backend_ptr->init_calls == 0);
        disabled.update(server_gpu_power_phase::prefill);
        assert(disabled_backend_ptr->set_calls == 0);
    }

    // 6. Test invalid power config rejected
    {
        auto             invalid_backend = std::make_unique<fake_gpu_power_backend>();
        server_gpu_power invalid(std::move(invalid_backend));
        assert(!invalid.init({ 201, 165, -1, -1, 0 }));
    }

    // 7. Test failing power backend
    {
        auto   failing_backend        = std::make_unique<fake_gpu_power_backend>();
        auto * failing_backend_ptr    = failing_backend.get();
        failing_backend->fail_on_call = 2;
        server_gpu_power failing(std::move(failing_backend));
        assert(failing.init({ 200, 165, -1, -1, 0 }));
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
    }

    // 8. Test memory overclock governor (target > 10501, e.g. 11001 MHz)
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        // 11001 MHz: max_stock=10501, base_p0=10251 -> offset=(11001-10251)*2 = 1500 MHz
        assert(governor.init({ -1, -1, 11001, -1, 0 }));
        assert(governor.enabled());
        assert(backend_ptr->init_calls == 1);
        assert(backend_ptr->set_mem_calls == 0);
        assert(backend_ptr->set_offset_calls == 0);

        governor.update(server_gpu_power_phase::prefill);
        assert(backend_ptr->set_mem_calls == 0);
        assert(backend_ptr->set_offset_calls == 0);

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);
        assert(backend_ptr->set_offset_calls == 1);
        assert(backend_ptr->applied_offsets.back() == 1500);

        // Deduplicated
        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->set_offset_calls == 1);

        // Idle resets both offset and locked clock
        governor.update(server_gpu_power_phase::idle);
        assert(backend_ptr->reset_offset_calls == 1);
        assert(backend_ptr->reset_mem_calls == 1);

        governor.shutdown();
        assert(backend_ptr->shutdown_calls == 1);
    }

    // 9. Test memory overclock clamp to MAX_SAFE_MEM_CLOCK_MHZ (11001)
    {
        auto             backend     = std::make_unique<fake_gpu_power_backend>();
        auto *           backend_ptr = backend.get();
        server_gpu_power governor(std::move(backend));

        // 12501 MHz exceeds MAX_SAFE_MEM_CLOCK_MHZ (11001), clamped to 11001
        assert(governor.init({ -1, -1, 12501, -1, 0 }));
        assert(governor.enabled());

        governor.update(server_gpu_power_phase::decode);
        assert(backend_ptr->set_mem_calls == 1);
        assert(backend_ptr->applied_mem_clocks.back() == 10501);
        assert(backend_ptr->set_offset_calls == 1);
        // Clamped to 11001: (11001 - 10251) * 2 = 1500
        assert(backend_ptr->applied_offsets.back() == 1500);

        governor.shutdown();
        assert(backend_ptr->reset_offset_calls == 1);
    }

    return 0;
}
