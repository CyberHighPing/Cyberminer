#include "panel_b.cuh"
#include <array>
#include <cassert>
#include <cstring>
#include <vector>

using pearl_jobgen::Size;
using pearl_panel_b::chunk;
using pearl_panel_b::panel_limit;

static pearl_jobgen::Plan make_plan(Size rows, unsigned k) {
    pearl_jobgen::Plan plan;
    assert(pearl_jobgen::make_plan(rows, k, plan));
    return plan;
}

static void geometry(Size rows, unsigned k) {
    const auto plan = make_plan(rows, k);
    const Size capacity = pearl_panel_b::panel_capacity(plan);
    Size whole_rows = std::min(rows, panel_limit / k);
    if (whole_rows < rows) {
        while (whole_rows * k % chunk) --whole_rows;
    }
    assert(capacity > 0 && capacity == whole_rows * k);
    Size next_leaf = 0;
    for (Size start = 0; start < plan.value_bytes;) {
        const Size used = std::min(capacity, plan.value_bytes - start);
        assert(start % chunk == 0 && start % k == 0 && used % k == 0);
        assert(start / chunk == next_leaf);
        next_leaf += (used + chunk - 1) / chunk;
        start += used;
    }
    assert(next_leaf == plan.value_chunks);
}

static void check_leaves(Size rows, unsigned k) {
    const auto plan = make_plan(rows, k);
    const Size capacity = pearl_panel_b::panel_capacity(plan);
    std::vector<std::uint8_t> raw(plan.value_bytes);
    for (Size i = 0; i < raw.size(); ++i) raw[i] = std::uint8_t(i * 29 + (i >> 7) + 3 * (i >> 19));
    std::array<std::uint8_t, 32> key_bytes{};
    for (unsigned i = 0; i < key_bytes.size(); ++i) key_bytes[i] = i;
    const auto key = pearl_jobgen::from_bytes(key_bytes.data());
    for (Size start = 0; start < raw.size(); start += capacity) {
        const Size used = std::min<Size>(capacity, raw.size() - start);
        const Size count = (used + chunk - 1) / chunk;
        for (Size local : std::array<Size,3>{0, count / 2, count - 1}) {
            const Size global = start / chunk + local;
            const auto panel = pearl_panel_b::value_leaf(raw.data() + start, used, local, global, key, plan.value_chunks == 1);
            const auto whole = pearl_jobgen::leaf<false>(raw.data(), raw.size(), global, key, plan.value_chunks == 1);
            assert(std::memcmp(&panel, &whole, sizeof(whole)) == 0);
        }
    }
}

int main() {
    for (unsigned k = 1024; k <= 65536; k += 32) {
        const Size max_rows = panel_limit / k;
        for (Size rows : std::array<Size,4>{1, max_rows, max_rows + 1, 2 * max_rows + 7}) geometry(rows, k);
    }
    assert(pearl_panel_b::panel_capacity(make_plan(65536,1152)) == 50328576);
    assert(pearl_panel_b::panel_capacity(make_plan(1044480,13312)) == 50319360);
    assert(pearl_panel_b::panel_capacity(make_plan(131072,16384)) == panel_limit);
    const auto huge_unit = (1u << 20) + 8;
    assert(pearl_panel_b::panel_capacity(make_plan(1,huge_unit)) == huge_unit);
    for (const auto& invalid : {pearl_jobgen::Plan{}, make_plan(64,huge_unit), make_plan(4,UINT32_MAX-7)}) {
        bool rejected = false;
        try { (void)pearl_panel_b::panel_capacity(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    check_leaves(1,1056);
    check_leaves(3 * 43688 + 1,1152);
    check_leaves(4097,13312);
    check_leaves(3073,16384);
}
