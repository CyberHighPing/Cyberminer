#pragma once

namespace pearl_geometry {
constexpr bool supported_scan_depth(int k) {
    switch(k){
        case 8192: case 10240: case 12288: case 13312: case 14336:
        case 16384: case 20480: case 24576: case 32768:
            return true;
        default:
            return false;
    }
}
}
