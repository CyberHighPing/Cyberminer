#include "geometry_filter.hpp"
#include "rect_vector_layout.hpp"
#include <array>

#include <cassert>
#include <cstdint>
#include <vector>

struct Ticket { std::uint64_t tile; };

int main() {
    // The rectangular fold reads all eight vectors of every 32-word ticket row.
    // Mirror the kernel's loader and consumer indices using a synthetic row/column tag.
    std::array<unsigned, 16 * 64> shared{};
    for (unsigned tid = 0; tid < 128; ++tid) {
        for (unsigned step = 0; step < 8; ++step) {
            const unsigned row = (tid >> 6) + 2 * step;
            const unsigned col = tid & 63;
            shared.at(pearl_rect_vector::shared_vector(row, col)) = 1 + row * 64 + col;
        }
    }
    for (unsigned tid = 0; tid < 128; ++tid) {
        const unsigned ticket = tid / 16, row = tid % 16;
        for (unsigned step = 0; step < 8; ++step) {
            const unsigned col = ticket * 8 + step;
            assert(shared.at(pearl_rect_vector::shared_vector(row, col)) == 1 + row * 64 + col);
        }
    }
    using namespace pearl_geometry;
    const int rows = 4096, cols = 16384;
    const int m = 18 * rows, n = 5 * cols;
    const std::size_t words = std::size_t(m / 16) * (n / 32) * 16;
    assert(expandable(true, true, m, n, 16384, 16, 32, rows, cols, 90, words));
    assert(!expandable(true, true, m - 1, n, 16384, 16, 32, rows, cols, 90, words));
    assert(!expandable(true, true, m, n, 8192, 16, 32, rows, cols, 90, words));
    assert(!expandable(true, false, m, n, 16384, 16, 32, rows, cols, 90, words));
    assert(!expandable(true, true, m, n, 16384, 16, 32, rows, cols, 90, legacy_words));

    constexpr std::size_t compact = 4096, panel = 16, full = 64, extra = 1024;
    constexpr auto fixed = compact + extra + (full - panel) * sizeof(std::uint32_t);
    assert(fits_full(compact, panel, full, extra, fixed + pearl_pool::reserve));
    assert(!fits_full(compact, panel, full, extra, fixed + pearl_pool::reserve - 1));
    assert(!fits_full(pearl_pool::ceiling, panel, full, 0, pearl_pool::ceiling + pearl_pool::reserve));

    constexpr int tile_rows = 4, tile_cols = 9, window_cols = 3;
    std::vector<Ticket> tickets;
    for (int id = 0; id < tile_rows * tile_cols; ++id)
        if (id % 4 != 1) tickets.push_back({static_cast<std::uint64_t>(id)});
    std::vector<std::uint64_t> expected, written;
    for (int first_col = 0; first_col < tile_cols; first_col += window_cols)
        for (int row = 0; row < tile_rows; ++row)
            for (int col = first_col; col < first_col + window_cols; ++col) {
                const int id = row * tile_cols + col;
                if (id % 4 != 1) expected.push_back(id);
            }
    append_window_order(tickets.data(), tickets.size(), tile_cols, window_cols,
                        [&](const Ticket* result, std::size_t count) {
                            for (std::size_t i = 0; i < count; ++i) written.push_back(result[i].tile);
                        });
    assert(written == expected);
}
