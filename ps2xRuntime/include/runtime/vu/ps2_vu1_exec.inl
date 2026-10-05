#pragma once

// VU1 upper/lower instruction semantics, header-inline so that VU1 microcode recompiled to C++
// (ps2_vu1_recomp) can call execUpper/execLower with constant instruction words and have the
// decode folded away. The interpreter (ps2_vu1_core.cpp) includes this same file, so both
// execute literally the same code.

#include "runtime/ps2_vu1.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_memory.h"

#include <cmath>
#include <cstring>
#include <limits>

// Instruction field extraction helpers
static inline uint8_t DEST(uint32_t i) { return (uint8_t)((i >> 21) & 0xF); }
static inline uint8_t FT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t FS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t FD(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t BC(uint32_t i) { return (uint8_t)(i & 0x3); }

// Lower instruction field helpers
static inline uint8_t LIT(uint32_t i) { return (uint8_t)((i >> 16) & 0x1F); }
static inline uint8_t LIS(uint32_t i) { return (uint8_t)((i >> 11) & 0x1F); }
static inline uint8_t LID(uint32_t i) { return (uint8_t)((i >> 6) & 0x1F); }
static inline uint8_t VIT(uint32_t i) { return (uint8_t)((i >> 16) & 0xF); }
static inline uint8_t VIS(uint32_t i) { return (uint8_t)((i >> 11) & 0xF); }
static inline uint8_t VID(uint32_t i) { return (uint8_t)((i >> 6) & 0xF); }
static inline int16_t IMM11(uint32_t i) { return (int16_t)(int32_t)((int32_t)(i << 21) >> 21); }
static inline int16_t IMM15(uint32_t i)
{
    uint32_t lo11 = i & 0x7FF;
    uint32_t hi4 = (i >> 21) & 0xF;
    uint32_t raw = (hi4 << 11) | lo11;
    return (int16_t)(int32_t)((int32_t)(raw << 17) >> 17);
}


namespace ps2x_vu1_exec_detail
{
    // normalizeOperand on four lanes at once (same results): denormals -> signed zero,
    // Inf/NaN -> signed max.
    __attribute__((always_inline)) inline void normalizeOperand4(const float *src, float *dst)
    {
        typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
        typedef int32_t i32x4 __attribute__((ext_vector_type(4)));
        u32x4 bits;
        std::memcpy(&bits, src, sizeof(bits));
        const u32x4 exponent = bits & 0x7F800000u;
        const u32x4 sign = bits & 0x80000000u;
        const u32x4 isZero = (u32x4)(i32x4)(exponent == 0u);
        const u32x4 isSpecial = isZero | (u32x4)(i32x4)(exponent == 0x7F800000u);
        const u32x4 special = (isZero & sign) | (~isZero & (sign | 0x7F7FFFFFu));
        const u32x4 out = (isSpecial & special) | (~isSpecial & bits);
        std::memcpy(dst, &out, sizeof(out));
    }

    constexpr uint8_t laneForComponent(uint32_t component)
    {
        return static_cast<uint8_t>(1u << (3u - component));
    }

    inline int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }
}

namespace ps2x_vu1_exec_detail
{
    inline float vuEatan(float value)
    {
        constexpr float coefficients[] = {
            0.999999344348907f,
            -0.333298563957214f,
            0.199465364217758f,
            -0.13085337519646f,
            0.096420042216778f,
            -0.055909886956215f,
            0.021861229091883f,
            -0.004054057877511f};
        constexpr float quarterPi = 0.785398185253143f;

        const float squared = value * value;
        float polynomial = coefficients[7];
        for (int index = 6; index >= 0; --index)
            polynomial = coefficients[index] + squared * polynomial;
        return quarterPi + value * polynomial;
    }

    inline float vuEsin(float value)
    {
        constexpr float coefficients[] = {
            1.0f,
            -0.166666567325592f,
            0.008333025500178f,
            -0.000198074136279f,
            0.000002601886990f};

        const float squared = value * value;
        float polynomial = coefficients[4];
        for (int index = 3; index >= 0; --index)
            polynomial = coefficients[index] + squared * polynomial;
        return value * polynomial;
    }

    inline float vuEexp(float value)
    {
        constexpr float coefficients[] = {
            0.249998688697815f,
            0.031257584691048f,
            0.002591371303424f,
            0.000171562001924f,
            0.000005430199963f,
            0.000000690600018f};

        float polynomial = coefficients[5];
        for (int index = 4; index >= 0; --index)
            polynomial = coefficients[index] + value * polynomial;
        polynomial = 1.0f + value * polynomial;
        polynomial *= polynomial;
        polynomial *= polynomial;
        return polynomial != 0.0f ? 1.0f / polynomial : std::numeric_limits<float>::max();
    }
}

using namespace ps2x_vu1_exec_detail;

// Hot helpers used by every instruction (header-inline so recompiled code can fold them).

inline void VU1Interpreter::commitDueFlags()
{
    while (m_liveFlag != 0u && m_flagPipeline[m_flagHead].readyCycle <= m_cycle)
    {
        FlagPipelineEntry &entry = m_flagPipeline[m_flagHead];

        if (entry.writesMac)
            m_state.mac = entry.mac;
        if (entry.writesStatus)
        {
            const uint32_t current = entry.status & 0xFu;
            m_state.status = (m_state.status & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
        }
        if (entry.writesSticky)
        {
            m_state.status = (m_state.status & 0x03Fu) | (entry.status & 0xFC0u);
        }
        if (entry.writesClip)
            m_state.clip = entry.clip;
        entry.valid = false;
        m_flagHead = (m_flagHead + 1u) % kMaxFlagEntries;
        --m_liveFlag;
    }
}

// Flag entries only drive the commit scheduler when they are committed eagerly.
inline uint64_t VU1Interpreter::flagReadyCycle()
{
    const uint64_t ready = m_cycle + kFmacLatency;
    return m_lazyFlags ? ready : notePipelineReady(ready);
}

inline VU1Interpreter::FlagPipelineEntry *VU1Interpreter::allocFlagEntry()
{
    if (m_liveFlag >= kMaxFlagEntries && m_lazyFlags)
        commitDueFlags();
    if (m_liveFlag >= kMaxFlagEntries)
        return nullptr;
    FlagPipelineEntry *entry = &m_flagPipeline[(m_flagHead + m_liveFlag) % kMaxFlagEntries];
    entry->mac = 0u;
    entry->status = 0u;
    entry->extraSticky = 0u;
    entry->clip = 0u;
    entry->writesMac = false;
    entry->writesStatus = false;
    entry->writesSticky = false;
    entry->writesClip = false;
    entry->valid = true;
    ++m_liveFlag;
    return entry;
}

#define PS2X_FOR_EACH_FLAG_ENTRY(e) \
    for (uint32_t _i = 0; _i < m_liveFlag; ++_i) \
        if (FlagPipelineEntry &e = m_flagPipeline[(m_flagHead + _i) % kMaxFlagEntries]; true)
inline __attribute__((always_inline)) float VU1Interpreter::normalizeOperand(float value) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Denormals flush to signed zero, Inf/NaN clamp to signed max (written as selects so
    // the compiler emits csel instead of branches).
    const uint32_t exponent = bits & 0x7F800000u;
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t special = exponent == 0u ? sign : (sign | 0x7F7FFFFFu);
    bits = (exponent == 0u || exponent == 0x7F800000u) ? special : bits;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline __attribute__((always_inline)) void VU1Interpreter::applyDest(float *dst, const float *result, uint8_t dest)
{
    if (dest & 0x8u)
        dst[0] = result[0];
    if (dest & 0x4u)
        dst[1] = result[1];
    if (dest & 0x2u)
        dst[2] = result[2];
    if (dest & 0x1u)
        dst[3] = result[3];
}

inline __attribute__((always_inline)) void VU1Interpreter::applyDestAcc(const float *result, uint8_t dest)
{
    applyDest(m_state.acc, result, dest);
}

template <bool Flags>
inline __attribute__((always_inline)) void VU1Interpreter::applyFmacDest(float *dst, float *result, uint8_t dest, uint32_t upper)
{
    if constexpr (Flags)
    {
        uint8_t laneFlags[4]{};
        normalizeFmacResult(result, dest, laneFlags, upper);
        updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest, upper));
    }
    else
        normalizeFmacValue(result, dest, upper);
    applyDest(dst, result, dest);
}

template <bool Flags>
inline __attribute__((always_inline)) void VU1Interpreter::applyFmacDestAcc(float *result, uint8_t dest, uint32_t upper)
{
    if constexpr (Flags)
    {
        uint8_t laneFlags[4]{};
        normalizeFmacResult(result, dest, laneFlags, upper);
        updateFmacFlags(laneFlags, dest, calculateFmacProductSticky(dest, upper));
    }
    else
        normalizeFmacValue(result, dest, upper);
    applyDestAcc(result, dest);
}

// Value-only variant of normalizeFmacResult for code whose flags are never observed. The host
// result is the exact value rounded toward zero once, so a lane whose exponent is in 1..0xFD
// needs no clamping; only zero/denormal and near-overflow lanes take the exact path.
inline __attribute__((always_inline)) void VU1Interpreter::normalizeFmacValue(float *result, uint8_t dest, uint32_t upper)
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        uint32_t bits;
        std::memcpy(&bits, &result[component], sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (__builtin_expect(exponent != 0u && exponent < 0xFEu, 1))
            continue;
        VuExact exactResult = 0.0;
        if (calculateFmacExactResult(component, exactResult, upper))
            (void)normalizeFmacExactResult(result[component], exactResult);
        else
        {
            uint32_t flags = 0u;
            result[component] = normalizeResult(result[component], flags);
        }
    }
}

inline __attribute__((always_inline)) float VU1Interpreter::broadcast(const float *vf, uint8_t bc)
{
    return normalizeOperand(vf[bc & 3u]);
}

inline __attribute__((always_inline)) bool VU1Interpreter::calculateFmacExactResult(uint32_t component,
                                               VuExact &result, uint32_t upper) const
{
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu
                                ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu))
                                : 0xFFu;
    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);

    const auto operand = [this](float value)
    {
        return static_cast<VuExact>(normalizeOperand(value));
    };
    const auto vs = [&](uint32_t lane)
    {
        return operand(m_state.vf[fs][lane]);
    };
    const auto vt = [&](uint32_t lane)
    {
        return operand(m_state.vf[ft][lane]);
    };
    const auto acc = [&](uint32_t lane)
    {
        return operand(m_state.acc[lane]);
    };

    const VuExact q = operand(m_state.q);
    const VuExact i = operand(m_state.i);

    if (op < 0x3Cu)
    {
        if (op <= 0x03u)
            result = vs(component) + vt(op & 3u);
        else if (op <= 0x07u)
            result = vs(component) - vt(op & 3u);
        else if (op <= 0x0Bu)
            result = acc(component) + vs(component) * vt(op & 3u);
        else if (op <= 0x0Fu)
            result = acc(component) - vs(component) * vt(op & 3u);
        else if (op >= 0x18u && op <= 0x1Bu)
            result = vs(component) * vt(op & 3u);
        else
        {
            switch (op)
            {
            case 0x1Cu:
                result = vs(component) * q;
                break;
            case 0x1Eu:
                result = vs(component) * i;
                break;
            case 0x20u:
                result = vs(component) + q;
                break;
            case 0x21u:
                result = acc(component) + vs(component) * q;
                break;
            case 0x22u:
                result = vs(component) + i;
                break;
            case 0x23u:
                result = acc(component) + vs(component) * i;
                break;
            case 0x24u:
                result = vs(component) - q;
                break;
            case 0x25u:
                result = acc(component) - vs(component) * q;
                break;
            case 0x26u:
                result = vs(component) - i;
                break;
            case 0x27u:
                result = acc(component) - vs(component) * i;
                break;
            case 0x28u:
                result = vs(component) + vt(component);
                break;
            case 0x29u:
                result = acc(component) + vs(component) * vt(component);
                break;
            case 0x2Au:
                result = vs(component) * vt(component);
                break;
            case 0x2Cu:
                result = vs(component) - vt(component);
                break;
            case 0x2Du:
                result = acc(component) - vs(component) * vt(component);
                break;
            case 0x2Eu:
            {
                static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
                static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
                result = component == 3u
                             ? 0.0
                             : acc(component) - vs(left[component]) * vt(right[component]);
                break;
            }
            default:
                return false;
            }
        }
        return true;
    }

    if (special <= 0x03u)
        result = vs(component) + vt(special & 3u);
    else if (special <= 0x07u)
        result = vs(component) - vt(special & 3u);
    else if (special <= 0x0Bu)
        result = acc(component) + vs(component) * vt(special & 3u);
    else if (special <= 0x0Fu)
        result = acc(component) - vs(component) * vt(special & 3u);
    else if (special >= 0x18u && special <= 0x1Bu)
        result = vs(component) * vt(special & 3u);
    else
    {
        switch (special)
        {
        case 0x1Cu:
            result = vs(component) * q;
            break;
        case 0x1Eu:
            result = vs(component) * i;
            break;
        case 0x20u:
            result = vs(component) + q;
            break;
        case 0x21u:
            result = acc(component) + vs(component) * q;
            break;
        case 0x22u:
            result = vs(component) + i;
            break;
        case 0x23u:
            result = acc(component) + vs(component) * i;
            break;
        case 0x24u:
            result = vs(component) - q;
            break;
        case 0x25u:
            result = acc(component) - vs(component) * q;
            break;
        case 0x26u:
            result = vs(component) - i;
            break;
        case 0x27u:
            result = acc(component) - vs(component) * i;
            break;
        case 0x28u:
            result = vs(component) + vt(component);
            break;
        case 0x29u:
            result = acc(component) + vs(component) * vt(component);
            break;
        case 0x2Au:
            result = vs(component) * vt(component);
            break;
        case 0x2Cu:
            result = vs(component) - vt(component);
            break;
        case 0x2Du:
            result = acc(component) - vs(component) * vt(component);
            break;
        case 0x2Eu:
        {
            static constexpr uint8_t left[4] = {1u, 2u, 0u, 3u};
            static constexpr uint8_t right[4] = {2u, 0u, 1u, 3u};
            result = component == 3u
                         ? 0.0
                         : vs(left[component]) * vt(right[component]);
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

inline __attribute__((always_inline)) uint32_t VU1Interpreter::calculateFmacProductSticky(uint8_t dest, uint32_t upper) const
{
    uint32_t extraSticky = 0u;
    const uint8_t op = static_cast<uint8_t>(upper & 0x3Fu);
    const uint8_t special = op >= 0x3Cu ? static_cast<uint8_t>((upper & 3u) | ((upper >> 4) & 0x7Cu)) : 0xFFu;
    const bool productSum =
        (op >= 0x08u && op <= 0x0Fu) ||
        op == 0x21u || op == 0x23u || op == 0x25u || op == 0x27u ||
        op == 0x29u || op == 0x2Du || op == 0x2Eu ||
        (special >= 0x08u && special <= 0x0Fu) ||
        special == 0x21u || special == 0x23u || special == 0x25u ||
        special == 0x27u || special == 0x29u || special == 0x2Du;
    if (!productSum)
        return 0u;

    const uint8_t fs = FS(upper);
    const uint8_t ft = FT(upper);
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((dest & laneForComponent(component)) == 0u)
            continue;
        static constexpr uint8_t crossLeft[4] = {1u, 2u, 0u, 3u};
        static constexpr uint8_t crossRight[4] = {2u, 0u, 1u, 3u};
        const uint8_t leftComponent = op == 0x2Eu ? crossLeft[component] : static_cast<uint8_t>(component);
        const float left = normalizeOperand(m_state.vf[fs][leftComponent]);
        float right = 0.0f;
        if ((op >= 0x08u && op <= 0x0Fu) || (special >= 0x08u && special <= 0x0Fu))
        {
            right = normalizeOperand(m_state.vf[ft][(op >= 0x08u && op <= 0x0Fu ? op : special) & 3u]);
        }
        else if (op == 0x21u || op == 0x25u || special == 0x21u || special == 0x25u)
        {
            right = normalizeOperand(m_state.q);
        }
        else if (op == 0x23u || op == 0x27u || special == 0x23u || special == 0x27u)
        {
            right = normalizeOperand(m_state.i);
        }
        else if (op == 0x2Eu)
        {
            right = normalizeOperand(m_state.vf[ft][crossRight[component]]);
        }
        else
        {
            right = normalizeOperand(m_state.vf[ft][component]);
        }

        float product = left * right;
        const VuExact exactProduct = static_cast<VuExact>(left) * static_cast<VuExact>(right);
        const uint8_t productFlags = normalizeFmacExactResult(product, exactProduct);
        // Product-sum instructions report Z/S/U/O from the add/subtract result
        // as current flags, while every product condition accumulates into the
        // corresponding sticky flag.
        extraSticky |= productFlags & 0xFu;
    }
    return extraSticky;
}

inline __attribute__((always_inline)) uint8_t VU1Interpreter::normalizeFmacExactResult(float &value,
                                                  VuExact exactResult) const
{
    const bool negative = std::signbit(exactResult);
    const VuExact magnitude = std::fabs(exactResult);
    const VuExact maximum = static_cast<VuExact>(std::numeric_limits<float>::max());
    const VuExact minimum = static_cast<VuExact>(std::numeric_limits<float>::min());
    uint8_t flags = negative ? 0x2u : 0u;

    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > maximum)
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < minimum)
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }

    return flags;
}

inline __attribute__((always_inline)) void VU1Interpreter::normalizeFmacResult(float *result, uint8_t dest,
                                         uint8_t laneFlags[4], uint32_t upper)
{
    for (uint32_t component = 0; component < 4u; ++component)
    {
        laneFlags[component] = 0u;
        if ((dest & laneForComponent(component)) == 0u)
            continue;

        VuExact exactResult = 0.0;
        if (calculateFmacExactResult(component, exactResult, upper))
        {
            laneFlags[component] = normalizeFmacExactResult(result[component], exactResult);
            continue;
        }

        uint32_t flags = 0u;
        result[component] = normalizeResult(result[component], flags);
        laneFlags[component] = static_cast<uint8_t>(flags);
    }
}

inline __attribute__((always_inline)) float VU1Interpreter::normalizeResult(float value, uint32_t &laneFlags) const
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t exponent = (bits >> 23) & 0xFFu;

    laneFlags = sign != 0u ? 0x2u : 0u;
    if (magnitude == 0u)
    {
        laneFlags |= 0x1u;
    }
    else if (exponent == 0u)
    {
        laneFlags |= 0x5u;
        bits = sign;
    }
    else if (exponent == 0xFFu)
    {
        laneFlags |= 0x8u;
        bits = sign | 0x7F7FFFFFu;
    }

    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline __attribute__((always_inline)) void VU1Interpreter::updateFmacFlags(const uint8_t laneFlags[4], uint8_t dest,
                                     uint32_t extraSticky)
{
    if (dest == 0u)
        return;

    uint32_t mac = 0u;
    uint32_t status = 0u;
    for (uint32_t component = 0; component < 4u; ++component)
    {
        const uint8_t lane = laneForComponent(component);
        if ((dest & lane) == 0u)
            continue;

        const uint32_t flags = laneFlags[component];
        if ((flags & 0x1u) != 0u)
            mac |= lane;
        if ((flags & 0x2u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 4;
        if ((flags & 0x4u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 8;
        if ((flags & 0x8u) != 0u)
            mac |= static_cast<uint32_t>(lane) << 12;
        status |= flags;
    }

    FlagPipelineEntry *entry = allocFlagEntry();
    if (!entry)
    {
        reportReservedInstruction(true, 0xFFFFFFFFu);
        return;
    }

    entry->issueCycle = m_cycle;
    entry->readyCycle = flagReadyCycle();
    entry->mac = mac;
    entry->status = status;
    entry->extraSticky = extraSticky;
    entry->writesMac = true;
    entry->writesStatus = true;
}

inline int32_t VU1Interpreter::readBranchVi(uint8_t reg) const
{
    if (reg == 0u)
        return 0;
    if (m_viBranchBackupValid &&
        m_viBranchBackupReg == reg)
    {
        return m_viBranchBackupValue;
    }
    return m_state.vi[reg];
}

inline void VU1Interpreter::recordViWriteForBranch(uint8_t reg, int32_t oldValue)
{
    if (reg == 0u)
        return;
    m_viBranchBackupValue = oldValue;
    m_viBranchBackupReg = reg;
    m_viBranchBackupValid = true;
}

inline uint32_t VU1Interpreter::microAddressMask() const
{
    return m_unit == Unit::VU1 ? 0x3FFFu : 0x0FFFu;
}

inline void VU1Interpreter::advanceOneCycle()
{
    ++m_cycle;
    m_state.cycles = m_cycle;
    // LSU commits become visible at the cycle boundary before PATH1 consumes
    // its next qword from VU memory.
    if (m_cycle >= m_nextCommitCycle)
        commitReadyPipelines();
    if (m_xgkick.active)
        progressXgkick();
}

// Same result as calling advanceOneCycle() until targetCycle, but cycles before the next
// pipeline commit are skipped in one step: nothing can change VU memory in between, so PATH1
// just gets their transfer credit at once.
inline void VU1Interpreter::advanceTo(uint64_t targetCycle)
{
    while (m_cycle < targetCycle)
    {
        uint64_t stop = targetCycle;
        if (m_nextCommitCycle > m_cycle && m_nextCommitCycle < stop)
            stop = m_nextCommitCycle;
        if (stop - m_cycle > 1u)
        {
            const uint64_t skip = stop - m_cycle - 1u;
            m_cycle += skip;
            m_state.cycles = m_cycle;
            progressXgkick(static_cast<uint32_t>(skip));
        }
        advanceOneCycle();
    }
}

// Cycles until PATH1 next does something observable (reads a GIFtag or finishes).
inline uint64_t VU1Interpreter::cyclesUntilXgkickEvent() const
{
    if (!m_xgkick.active)
        return UINT64_MAX;
    const uint64_t qwords = m_xgkick.currentTagEnd == 0u
                                ? 1u
                                : (m_xgkick.currentTagEnd > m_xgkick.copiedBytes ? (m_xgkick.currentTagEnd - m_xgkick.copiedBytes + 15u) / 16u : 1u);
    const uint64_t need = qwords * 2u;
    return need > m_xgkick.cycleCredit ? need - m_xgkick.cycleCredit : 1u;
}

inline void VU1Interpreter::commitReadyPipelines()
{
    if (m_cycle < m_nextCommitCycle)
        return;
    // Recompiled code defers flag commits (m_lazyFlags) to the points where flags are observed;
    // FDIV also rewrites status, so due flags are committed before it to keep their order.
    if (!m_lazyFlags || (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle))
        commitDueFlags();

    if (m_fdiv.valid && m_fdiv.readyCycle <= m_cycle)
    {
        m_state.q = m_fdiv.value;
        const uint32_t currentDi = m_fdiv.statusDi & 0x30u;
        m_state.status = (m_state.status & 0xFCFu) | currentDi | (currentDi << 6);
        m_fdiv = {};
    }

    if (m_liveEfu) for (ScalarPipelineEntry &entry : m_efu)
    {
        if (entry.valid && entry.readyCycle <= m_cycle)
        {
            m_state.p = entry.value;
            entry = {};
            --m_liveEfu;
        }
    }

    if (m_liveStore) for (PendingStore &store : m_storePipeline)
    {
        if (!store.valid || store.readyCycle > m_cycle)
            continue;
        applyStore(store.address, store.words.data(), store.laneMask);
        store = {};
        --m_liveStore;
    }

    if (m_liveVf) for (PendingVfWrite &write : m_vfWritePipeline)
    {
        if (!write.valid || write.readyCycle > m_cycle)
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_vfLatestWrite[write.reg][component] == write.sequence)
            {
                m_state.vf[write.reg][component] = write.value[component];
            }
        }
        write = {};
        --m_liveVf;
    }

    if (m_liveVi) for (PendingViWrite &write : m_viWritePipeline)
    {
        if (!write.valid || write.readyCycle > m_cycle)
            continue;
        if (m_viLatestWrite[write.reg] == write.sequence)
            m_state.vi[write.reg] = static_cast<int16_t>(write.value);
        write = {};
        --m_liveVi;
    }

    if (m_liveAcc) for (PendingAccWrite &write : m_accWritePipeline)
    {
        if (!write.valid || write.readyCycle > m_cycle)
            continue;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((write.laneMask & laneForComponent(component)) != 0u &&
                m_accLatestWrite[component] == write.sequence)
            {
                m_state.acc[component] = write.value[component];
            }
        }
        write = {};
        --m_liveAcc;
    }

    // Recompute the earliest pending entry (everything still valid is in the future).
    uint64_t next = UINT64_MAX;
    auto consider = [&](bool valid, uint64_t ready) { if (valid && ready < next) next = ready; };
    if (m_liveFlag && !m_lazyFlags) consider(true, m_flagPipeline[m_flagHead].readyCycle);
    consider(m_fdiv.valid, m_fdiv.readyCycle);
    if (m_liveEfu) for (const auto &e : m_efu) consider(e.valid, e.readyCycle);
    if (m_liveStore) for (const auto &e : m_storePipeline) consider(e.valid, e.readyCycle);
    if (m_liveVf) for (const auto &e : m_vfWritePipeline) consider(e.valid, e.readyCycle);
    if (m_liveVi) for (const auto &e : m_viWritePipeline) consider(e.valid, e.readyCycle);
    if (m_liveAcc) for (const auto &e : m_accWritePipeline) consider(e.valid, e.readyCycle);
    m_nextCommitCycle = next;
}

inline void VU1Interpreter::progressXgkick(uint32_t cycles)
{
    if (!m_xgkick.active || !m_activeVuData || m_activeVuDataSize == 0u)
        return;

    m_xgkick.cycleCredit += cycles;
    while (m_xgkick.active && m_xgkick.cycleCredit >= 2u)
    {
        m_xgkick.cycleCredit -= 2u;
        if (m_xgkick.copiedBytes > XgkickPipeline::kBufferSize - 16u)
        {
            reportReservedInstruction(false, 0xFFFFFFFBu);
            m_xgkick.active = false;
            return;
        }

        const uint32_t qwordOffset = m_xgkick.copiedBytes;
        const uint32_t firstSource = (m_xgkick.sourceAddress + m_xgkick.copiedBytes) % m_activeVuDataSize;
        if (firstSource + 16u <= m_activeVuDataSize)
            std::memcpy(m_xgkick.packet.data() + m_xgkick.copiedBytes, m_activeVuData + firstSource, 16u);
        else
            for (uint32_t i = 0; i < 16u; ++i)
            {
                const uint32_t source = (m_xgkick.sourceAddress + m_xgkick.copiedBytes + i) % m_activeVuDataSize;
                m_xgkick.packet[m_xgkick.copiedBytes + i] = m_activeVuData[source];
            }
        m_xgkick.copiedBytes += 16u;

        if (m_xgkick.currentTagEnd == 0u)
        {
            uint64_t tagLo = 0;
            std::memcpy(&tagLo, m_xgkick.packet.data() + qwordOffset, sizeof(tagLo));
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint32_t format = static_cast<uint32_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            uint64_t tagBytes = 16u;
            if (format == 0u)
                tagBytes += static_cast<uint64_t>(nloop) * nreg * 16u;
            else if (format == 1u)
                tagBytes += ((static_cast<uint64_t>(nloop) * nreg + 1u) & ~1ull) * 8u;
            else if (format == 2u)
                tagBytes += static_cast<uint64_t>(nloop) * 16u;
            else
            {
                reportReservedInstruction(false, 0xFFFFFFF8u);
                m_xgkick.active = false;
                return;
            }

            if (tagBytes > XgkickPipeline::kBufferSize - qwordOffset)
            {
                reportReservedInstruction(false, 0xFFFFFFFBu);
                m_xgkick.active = false;
                return;
            }
            m_xgkick.currentTagEnd = qwordOffset + static_cast<uint32_t>(tagBytes);
            m_xgkick.currentTagEop = ((tagLo >> 15) & 1u) != 0u;
            if (m_xgkick.currentTagEop)
                m_xgkick.totalBytes = m_xgkick.currentTagEnd;
        }

        if (m_xgkick.copiedBytes >= m_xgkick.currentTagEnd)
        {
            if (m_xgkick.currentTagEop)
                finishXgkick();
            else
            {
                // The next transferred qword is another GIFtag.
                m_xgkick.currentTagEnd = 0u;
                m_xgkick.currentTagEop = false;
            }
        }
    }
}

inline void VU1Interpreter::queueVfWrite(uint8_t reg, uint8_t laneMask,
                                  const float value[4], uint32_t latency)
{
    if (reg == 0u || laneMask == 0u)
        return;
    for (PendingVfWrite &write : m_vfWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true; ++m_liveVf;
            write.readyCycle = notePipelineReady(m_cycle + latency);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_vfLatestWrite[reg][component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF7u);
}

inline void VU1Interpreter::queueViWrite(uint8_t reg, int32_t value, uint32_t latency)
{
    if (reg == 0u)
        return;
    for (PendingViWrite &write : m_viWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true; ++m_liveVi;
            write.readyCycle = notePipelineReady(m_cycle + latency);
            write.sequence = ++m_nextWriteSequence;
            write.reg = reg;
            write.value = value;
            m_viLatestWrite[reg] = write.sequence;
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF6u);
}

inline void VU1Interpreter::queueAccWrite(uint8_t laneMask, const float value[4], uint32_t latency)
{
    if (laneMask == 0u)
        return;
    for (PendingAccWrite &write : m_accWritePipeline)
    {
        if (!write.valid)
        {
            write = {};
            write.valid = true; ++m_liveAcc;
            write.readyCycle = notePipelineReady(m_cycle + latency);
            write.sequence = ++m_nextWriteSequence;
            write.laneMask = laneMask;
            std::copy(value, value + 4, write.value.begin());
            for (uint32_t component = 0; component < 4u; ++component)
            {
                if ((laneMask & laneForComponent(component)) != 0u)
                    m_accLatestWrite[component] = write.sequence;
            }
            return;
        }
    }
    reportReservedInstruction(true, 0xFFFFFFF5u);
}

inline void VU1Interpreter::queueClip(uint32_t clip)
{
    m_workingClip = ((m_workingClip << 6) | (clip & 0x3Fu)) & 0xFFFFFFu;
    if (FlagPipelineEntry *entry = allocFlagEntry())
    {
        entry->issueCycle = m_cycle;
        entry->readyCycle = flagReadyCycle();
        entry->clip = m_workingClip;
        entry->writesClip = true;
        return;
    }
    reportReservedInstruction(true, 0xFFFFFFFDu);
}

inline void VU1Interpreter::queueP(float value, uint32_t latency)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    for (ScalarPipelineEntry &entry : m_efu)
    {
        if (!entry.valid)
        {
            entry.valid = true; ++m_liveEfu;
            entry.readyCycle = notePipelineReady(m_cycle + latency);
            entry.value = value;
            // EFU throughput is one cycle shorter than result visibility.
            m_efuResourceReady = m_cycle + (latency > 0u ? latency - 1u : 0u);
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFF9u);
}

inline void VU1Interpreter::queueQ(float value, uint32_t latency, uint32_t statusDi)
{
    uint32_t ignoredFlags = 0u;
    value = normalizeResult(value, ignoredFlags);
    m_fdiv.valid = true;
    m_fdiv.readyCycle = notePipelineReady(m_cycle + latency);
    m_fdiv.value = value;
    m_fdiv.statusDi = statusDi & 0x30u;
}

inline __attribute__((always_inline)) void VU1Interpreter::applyStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    if (m_activeVuData && address + 16u <= m_activeVuDataSize)
    {
        uint32_t oldWords[4]{};
        std::memcpy(oldWords, m_activeVuData + address, sizeof(oldWords));
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((laneMask & laneForComponent(component)) != 0u)
                oldWords[component] = words[component];
        }
        std::memcpy(m_activeVuData + address, oldWords, sizeof(oldWords));
    }
}

inline __attribute__((always_inline)) void VU1Interpreter::queueStore(uint32_t address, const uint32_t words[4], uint8_t laneMask)
{
    // Recompiled code: a store lands at the next cycle boundary, before anything (the next
    // pair, PATH1) can read VU memory, so writing it now is equivalent.
    if (m_lazyFlags)
    {
        applyStore(address, words, laneMask);
        return;
    }
    for (PendingStore &store : m_storePipeline)
    {
        if (!store.valid)
        {
            store.valid = true; ++m_liveStore;
            store.readyCycle = notePipelineReady(m_cycle + 1u);
            store.address = address;
            store.laneMask = laneMask;
            std::copy(words, words + 4, store.words.begin());
            return;
        }
    }
    reportReservedInstruction(false, 0xFFFFFFFCu);
}

inline void VU1Interpreter::queueFsset(uint16_t immediate)
{
    PS2X_FOR_EACH_FLAG_ENTRY(entry)
    {
        if (entry.issueCycle == m_cycle)
            entry.writesStatus = false;
    }

    if (FlagPipelineEntry *entry = allocFlagEntry())
    {
        entry->issueCycle = m_cycle;
        entry->readyCycle = flagReadyCycle();
        entry->status = static_cast<uint32_t>(immediate) & 0xFC0u;
        entry->writesSticky = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFEu);
}

inline void VU1Interpreter::queueFcset(uint32_t clip)
{
    m_workingClip = clip & 0xFFFFFFu;
    PS2X_FOR_EACH_FLAG_ENTRY(entry)
    {
        if (entry.issueCycle == m_cycle)
            entry.writesClip = false;
    }
    if (FlagPipelineEntry *entry = allocFlagEntry())
    {
        entry->issueCycle = m_cycle;
        entry->readyCycle = flagReadyCycle();
        entry->clip = m_workingClip;
        entry->writesClip = true;
        return;
    }
    reportReservedInstruction(false, 0xFFFFFFFAu);
}

inline void VU1Interpreter::startXgkick(uint32_t qwordAddress)
{
    if (m_unit != Unit::VU1 || !m_activeVuData || m_activeVuDataSize < 16u)
        return;

    const uint32_t sourceAddress = (qwordAddress * 16u) % m_activeVuDataSize;
    m_xgkick.clear();
    m_xgkick.active = true;
    m_xgkick.sourceAddress = sourceAddress;
    m_xgkick.cycleCredit = 1u; // XGKICK's issue cycle counts toward PATH1.
    m_xgkick.issueCycle = m_cycle;
}

inline void VU1Interpreter::finishXgkick()
{
    if (!m_xgkick.active)
        return;

    if (m_xgkickSink)
        m_xgkickSink(m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeMemory)
        m_activeMemory->submitGifPacket(GifPathId::Path1, m_xgkick.packet.data(), m_xgkick.totalBytes);
    else if (m_activeGs)
        m_activeGs->processGIFPacket(m_xgkick.packet.data(), m_xgkick.totalBytes);
    m_xgkick.active = false;
}

// CLIP judgement of vf[fs].xyz against |vf[ft].w| (six flag bits, as queued into the clip flag).
inline __attribute__((always_inline)) uint32_t VU1Interpreter::clipFlags(const float *vs, const float *vt)
{
    uint32_t wBits = 0u;
    std::memcpy(&wBits, &vt[3], sizeof(wBits));
    const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

    const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        bits ^= signMask;
        int32_t orderedBits = 0;
        std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
        return orderedBits > limit;
    };

    uint32_t flags = 0u;
    if (exceedsClipPlane(vs[0], 0x00000000u))
        flags |= 0x01u;
    if (exceedsClipPlane(vs[0], 0x80000000u))
        flags |= 0x02u;
    if (exceedsClipPlane(vs[1], 0x00000000u))
        flags |= 0x04u;
    if (exceedsClipPlane(vs[1], 0x80000000u))
        flags |= 0x08u;
    if (exceedsClipPlane(vs[2], 0x00000000u))
        flags |= 0x10u;
    if (exceedsClipPlane(vs[2], 0x80000000u))
        flags |= 0x20u;
    return flags;
}

// DIV / SQRT / RSQRT (lower special 0x38..0x3A): the value Q receives (already normalized, as
// queueQ stores it), the status D/I bits it sets and its latency.
inline __attribute__((always_inline)) void VU1Interpreter::fdivResult(uint32_t instr, float &out, uint32_t &outStatusDi,
                                                                      uint32_t &outLatency) const
{
    const uint8_t vfT = FT(instr);
    const uint8_t vfS = FS(instr);
    const uint8_t funct2 = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
    uint32_t ignoredFlags = 0u;
    switch (funct2)
    {
    case 0x38: // DIV
    {
        int fsf = (instr >> 21) & 0x3;
        int ftf = (instr >> 23) & 0x3;
        const float num = normalizeOperand(m_state.vf[vfS][fsf]);
        const float den = normalizeOperand(m_state.vf[vfT][ftf]);
        uint32_t statusDi = 0u;
        float result = 0.0f;
        if (den == 0.0f)
        {
            statusDi = num == 0.0f ? 0x10u : 0x20u;
            result = std::signbit(num) != std::signbit(den)
                         ? -std::numeric_limits<float>::max()
                         : std::numeric_limits<float>::max();
        }
        else
        {
            result = num / den;
        }
        out = normalizeResult(result, ignoredFlags);
        outStatusDi = statusDi & 0x30u;
        outLatency = 7u;
        return;
    }
    case 0x39: // SQRT
    {
        int ftf = (instr >> 23) & 0x3;
        const float val = normalizeOperand(m_state.vf[vfT][ftf]);
        out = normalizeResult(std::sqrt(std::fabs(val)), ignoredFlags);
        outStatusDi = val < 0.0f ? 0x10u : 0u;
        outLatency = 7u;
        return;
    }
    default: // 0x3A RSQRT
    {
        int fsf = (instr >> 21) & 0x3;
        int ftf = (instr >> 23) & 0x3;
        const float num = normalizeOperand(m_state.vf[vfS][fsf]);
        const float radicand = normalizeOperand(m_state.vf[vfT][ftf]);
        const float den = std::sqrt(std::fabs(radicand));
        uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
        float result = 0.0f;
        if (den != 0.0f)
            result = num / den;
        else
        {
            statusDi = num == 0.0f ? 0x10u : 0x20u;
            result = std::signbit(num)
                         ? -std::numeric_limits<float>::max()
                         : std::numeric_limits<float>::max();
        }
        out = normalizeResult(result, ignoredFlags);
        outStatusDi = statusDi & 0x30u;
        outLatency = 13u;
        return;
    }
    }
}

// EFU ops (lower special 0x70..0x7A, 0x7C, 0x7D): the value P receives (normalized, as queueP
// stores it) and its latency.
inline __attribute__((always_inline)) void VU1Interpreter::efuResult(uint32_t instr, float &out, uint32_t &outLatency) const
{
    const uint8_t vfS = FS(instr);
    const uint8_t funct2 = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
    float value = 0.0f;
    uint32_t latency = 0u;
    switch (funct2)
    {
    case 0x70: // ESADD
    {
        const float x = normalizeOperand(m_state.vf[vfS][0]);
        const float y = normalizeOperand(m_state.vf[vfS][1]);
        const float z = normalizeOperand(m_state.vf[vfS][2]);
        value = x * x + y * y + z * z;
        latency = 11u;
        break;
    }
    case 0x71: // ERSADD
    {
        const float x = normalizeOperand(m_state.vf[vfS][0]);
        const float y = normalizeOperand(m_state.vf[vfS][1]);
        const float z = normalizeOperand(m_state.vf[vfS][2]);
        const float sum = x * x + y * y + z * z;
        value = sum != 0.0f ? 1.0f / sum : sum;
        latency = 18u;
        break;
    }
    case 0x72: // ELENG
    {
        const float x = normalizeOperand(m_state.vf[vfS][0]);
        const float y = normalizeOperand(m_state.vf[vfS][1]);
        const float z = normalizeOperand(m_state.vf[vfS][2]);
        value = std::sqrt(x * x + y * y + z * z);
        latency = 18u;
        break;
    }
    case 0x73: // ERLENG
    {
        const float x = normalizeOperand(m_state.vf[vfS][0]);
        const float y = normalizeOperand(m_state.vf[vfS][1]);
        const float z = normalizeOperand(m_state.vf[vfS][2]);
        const float len = std::sqrt(x * x + y * y + z * z);
        value = len != 0.0f ? 1.0f / len : len;
        latency = 24u;
        break;
    }
    case 0x74: // EATANxy
    {
        const float x = normalizeOperand(m_state.vf[vfS][0]);
        const float y = normalizeOperand(m_state.vf[vfS][1]);
        value = x != 0.0f ? vuEatan(y / x) : 0.0f;
        latency = 54u;
        break;
    }
    case 0x75: // EATANxz
    {
        const float x = normalizeOperand(m_state.vf[vfS][0]);
        const float z = normalizeOperand(m_state.vf[vfS][2]);
        value = x != 0.0f ? vuEatan(z / x) : 0.0f;
        latency = 54u;
        break;
    }
    case 0x76: // ESUM
    {
        float sum = 0.0f;
        for (uint32_t component = 0; component < 4u; ++component)
            sum += normalizeOperand(m_state.vf[vfS][component]);
        value = sum;
        latency = 12u;
        break;
    }
    case 0x77: // ERSQRT
    {
        const uint32_t component = (instr >> 21) & 3u;
        const float v = normalizeOperand(m_state.vf[vfS][component]);
        float result = v;
        if (result >= 0.0f)
        {
            result = std::sqrt(result);
            if (result != 0.0f)
                result = 1.0f / result;
        }
        value = result;
        latency = 18u;
        break;
    }
    case 0x78: // ESQRT
    {
        const uint32_t component = (instr >> 21) & 3u;
        const float v = normalizeOperand(m_state.vf[vfS][component]);
        value = v >= 0.0f ? std::sqrt(v) : v;
        latency = 12u;
        break;
    }
    case 0x79: // ESIN
    {
        const uint32_t component = (instr >> 21) & 3u;
        value = vuEsin(normalizeOperand(m_state.vf[vfS][component]));
        latency = 29u;
        break;
    }
    case 0x7A: // ERCPR
    {
        const uint32_t component = (instr >> 21) & 3u;
        const float v = normalizeOperand(m_state.vf[vfS][component]);
        value = v != 0.0f ? 1.0f / v : v;
        latency = 12u;
        break;
    }
    case 0x7C: // EATAN
    {
        const uint32_t component = (instr >> 21) & 3u;
        value = vuEatan(normalizeOperand(m_state.vf[vfS][component]));
        latency = 54u;
        break;
    }
    default: // 0x7D EEXP
    {
        const uint32_t component = (instr >> 21) & 3u;
        value = vuEexp(normalizeOperand(m_state.vf[vfS][component]));
        latency = 44u;
        break;
    }
    }
    uint32_t ignoredFlags = 0u;
    out = normalizeResult(value, ignoredFlags);
    outLatency = latency;
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
template <bool Flags>
inline void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    float *vd = m_state.vf[fd];
    float normalizedVs[4];
    float normalizedVt[4];
    float normalizedAcc[4];
    ps2x_vu1_exec_detail::normalizeOperand4(m_state.vf[fs], normalizedVs);
    ps2x_vu1_exec_detail::normalizeOperand4(m_state.vf[ft], normalizedVt);
    ps2x_vu1_exec_detail::normalizeOperand4(m_state.acc, normalizedAcc);
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * bc;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * bc;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + q;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * q;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + i;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * i;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - q;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * q;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - i;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * i;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * vt[c];
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * vt[c];
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x2E: // OPMSUB
        result[0] = acc[0] - vs[1] * vt[2];
        result[1] = acc[1] - vs[2] * vt[0];
        result[2] = acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDest<Flags>(vd, result, dest, instr);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * bc;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * bc;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x1F: // CLIP
        {
            const uint32_t flags = clipFlags(m_state.vf[fs], m_state.vf[ft]);
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + q;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * q;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + i;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * i;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - q;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * q;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - i;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * i;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * vt[c];
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * vt[c];
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestAcc<Flags>(result, dest, instr);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}


// ============================================================================
// Fast upper path (recompiled code, "fast" mode): the FMAC ops as 4-lane SIMD with the host's
// round-toward-zero and flush-to-zero modes plus clamping to +-FLT_MAX, like PCSX2's microVU.
// Not bit-exact with the interpreter's exact float model, but the same values to within the
// last bit or so; only used where no MAC/status reader can see the flags.
// ============================================================================
namespace ps2x_vu1_fast
{
    typedef float f4 __attribute__((ext_vector_type(4)));
    typedef int32_t i4 __attribute__((ext_vector_type(4)));

    __attribute__((always_inline)) inline f4 clampv(f4 v)
    {
        return __builtin_elementwise_min(__builtin_elementwise_max(v, (f4)(-3.402823466e+38f)), (f4)(3.402823466e+38f));
    }
    __attribute__((always_inline)) inline f4 load(const float *p)
    {
        f4 v;
        std::memcpy(&v, p, sizeof(v));
        return clampv(v);
    }
    __attribute__((always_inline)) inline f4 loadRaw(const float *p)
    {
        f4 v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    __attribute__((always_inline)) inline void store(float *dst, f4 value, uint8_t dest)
    {
        if (dest == 0xFu)
        {
            std::memcpy(dst, &value, sizeof(value));
            return;
        }
        const i4 mask = {(dest & 8u) ? -1 : 0, (dest & 4u) ? -1 : 0, (dest & 2u) ? -1 : 0, (dest & 1u) ? -1 : 0};
        i4 oldBits, newBits;
        std::memcpy(&oldBits, dst, sizeof(oldBits));
        std::memcpy(&newBits, &value, sizeof(newBits));
        const i4 out = (newBits & mask) | (oldBits & ~mask);
        std::memcpy(dst, &out, sizeof(out));
    }
    __attribute__((always_inline)) inline f4 splat(float v) { return (f4)(v); }
}

template <bool Flags>
__attribute__((always_inline)) inline bool VU1Interpreter::execUpperFast(uint32_t instr)
{
    using namespace ps2x_vu1_fast;
    if constexpr (Flags)
        return false;
    const uint8_t dest = DEST(instr);
    const uint8_t ft = FT(instr), fs = FS(instr), fd = FD(instr);
    const uint8_t op = instr & 0x3Fu;
    // (always_inline: recompiled code calls this with a constant word and needs it all folded)
    auto vs = [&]() __attribute__((always_inline)) { return load(m_state.vf[fs]); };
    auto vt = [&]() __attribute__((always_inline)) { return load(m_state.vf[ft]); };
    auto acc = [&]() __attribute__((always_inline)) { return load(m_state.acc); };
    auto bc = [&](uint32_t c) __attribute__((always_inline)) { return splat(std::clamp(m_state.vf[ft][c & 3u], -3.402823466e+38f, 3.402823466e+38f)); };
    const f4 q = splat(std::clamp(m_state.q, -3.402823466e+38f, 3.402823466e+38f));
    const f4 i = splat(std::clamp(m_state.i, -3.402823466e+38f, 3.402823466e+38f));
    auto toFd = [&](f4 r) __attribute__((always_inline)) { store(m_state.vf[fd], clampv(r), dest); };
    auto toAcc = [&](f4 r) __attribute__((always_inline)) { store(m_state.acc, clampv(r), dest); };
    auto opRotate = [&](f4 a, f4 b) __attribute__((always_inline)) { return (f4){a.y, a.z, a.x, 0.0f} * (f4){b.z, b.x, b.y, 0.0f}; };

    if (op < 0x3Cu)
    {
        if (fd == 0u && op != 0x2Eu)
            return false; // writes to VF0 are discarded; let the exact path handle the oddity
        switch (op)
        {
        case 0x00: case 0x01: case 0x02: case 0x03: toFd(vs() + bc(op)); return true;
        case 0x04: case 0x05: case 0x06: case 0x07: toFd(vs() - bc(op)); return true;
        case 0x08: case 0x09: case 0x0A: case 0x0B: toFd(acc() + vs() * bc(op)); return true;
        case 0x0C: case 0x0D: case 0x0E: case 0x0F: toFd(acc() - vs() * bc(op)); return true;
        case 0x10: case 0x11: case 0x12: case 0x13: store(m_state.vf[fd], __builtin_elementwise_max(vs(), bc(op)), dest); return true;
        case 0x14: case 0x15: case 0x16: case 0x17: store(m_state.vf[fd], __builtin_elementwise_min(vs(), bc(op)), dest); return true;
        case 0x18: case 0x19: case 0x1A: case 0x1B: toFd(vs() * bc(op)); return true;
        case 0x1C: toFd(vs() * q); return true;
        case 0x1D: store(m_state.vf[fd], __builtin_elementwise_max(vs(), i), dest); return true;
        case 0x1E: toFd(vs() * i); return true;
        case 0x1F: store(m_state.vf[fd], __builtin_elementwise_min(vs(), i), dest); return true;
        case 0x20: toFd(vs() + q); return true;
        case 0x21: toFd(acc() + vs() * q); return true;
        case 0x22: toFd(vs() + i); return true;
        case 0x23: toFd(acc() + vs() * i); return true;
        case 0x24: toFd(vs() - q); return true;
        case 0x25: toFd(acc() - vs() * q); return true;
        case 0x26: toFd(vs() - i); return true;
        case 0x27: toFd(acc() - vs() * i); return true;
        case 0x28: toFd(vs() + vt()); return true;
        case 0x29: toFd(acc() + vs() * vt()); return true;
        case 0x2A: toFd(vs() * vt()); return true;
        case 0x2B: store(m_state.vf[fd], __builtin_elementwise_max(vs(), vt()), dest); return true;
        case 0x2C: toFd(vs() - vt()); return true;
        case 0x2D: toFd(acc() - vs() * vt()); return true;
        case 0x2E: // OPMSUB
            if (fd == 0u)
                return false;
            toFd((f4){acc().x, acc().y, acc().z, 0.0f} - opRotate(vs(), vt()));
            return true;
        case 0x2F: store(m_state.vf[fd], __builtin_elementwise_min(vs(), vt()), dest); return true;
        default: return false;
        }
    }

    const uint8_t sop = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
    const bool toFt = (sop >= 0x10u && sop <= 0x17u) || sop == 0x1Du;
    if (toFt && ft == 0u)
        return false;
    switch (sop)
    {
    case 0x00: case 0x01: case 0x02: case 0x03: toAcc(vs() + bc(sop)); return true;
    case 0x04: case 0x05: case 0x06: case 0x07: toAcc(vs() - bc(sop)); return true;
    case 0x08: case 0x09: case 0x0A: case 0x0B: toAcc(acc() + vs() * bc(sop)); return true;
    case 0x0C: case 0x0D: case 0x0E: case 0x0F: toAcc(acc() - vs() * bc(sop)); return true;
    case 0x10: case 0x11: case 0x12: case 0x13: // ITOF0/4/12/15
    {
        static constexpr float kScale[4] = {1.0f, 1.0f / 16.0f, 1.0f / 4096.0f, 1.0f / 32768.0f};
        i4 iv;
        std::memcpy(&iv, m_state.vf[fs], sizeof(iv));
        store(m_state.vf[ft], __builtin_convertvector(iv, f4) * splat(kScale[sop & 3u]), dest);
        return true;
    }
    case 0x14: case 0x15: case 0x16: case 0x17: // FTOI0/4/12/15 (truncate, saturate)
    {
        static constexpr float kScale[4] = {1.0f, 16.0f, 4096.0f, 32768.0f};
        const f4 scaled = vs() * splat(kScale[sop & 3u]);
        const i4 iv = __builtin_convertvector(scaled, i4);
        f4 bits;
        std::memcpy(&bits, &iv, sizeof(bits));
        i4 sat = iv;
        // Saturate like the VU (C conversion of out-of-range floats is undefined).
        for (int c = 0; c < 4; ++c)
            if (!(scaled[c] < 2147483648.0f && scaled[c] >= -2147483648.0f))
                sat[c] = scaled[c] < 0.0f ? INT32_MIN : INT32_MAX;
        std::memcpy(&bits, &sat, sizeof(bits));
        store(m_state.vf[ft], bits, dest);
        return true;
    }
    case 0x18: case 0x19: case 0x1A: case 0x1B: toAcc(vs() * bc(sop)); return true;
    case 0x1C: toAcc(vs() * q); return true;
    case 0x1D: store(m_state.vf[ft], __builtin_elementwise_abs(loadRaw(m_state.vf[fs])), dest); return true;
    case 0x1E: toAcc(vs() * i); return true;
    case 0x20: toAcc(vs() + q); return true;
    case 0x21: toAcc(acc() + vs() * q); return true;
    case 0x22: toAcc(vs() + i); return true;
    case 0x23: toAcc(acc() + vs() * i); return true;
    case 0x24: toAcc(vs() - q); return true;
    case 0x25: toAcc(acc() - vs() * q); return true;
    case 0x26: toAcc(vs() - i); return true;
    case 0x27: toAcc(acc() - vs() * i); return true;
    case 0x28: toAcc(vs() + vt()); return true;
    case 0x29: toAcc(acc() + vs() * vt()); return true;
    case 0x2A: toAcc(vs() * vt()); return true;
    case 0x2C: toAcc(vs() - vt()); return true;
    case 0x2D: toAcc(acc() - vs() * vt()); return true;
    case 0x2E: toAcc(opRotate(vs(), vt())); return true; // OPMULA
    case 0x2F: case 0x30: return true;                   // NOP
    default: return false;                               // CLIP etc.: exact path
    }
}

inline void VU1Interpreter::execLower(uint32_t instr, uint8_t *vuData, uint32_t dataSize, GS &gs, PS2Memory *memory, uint32_t upperInstr)
{
    (void)upperInstr;
    if (instr == 0x00000000 || instr == 0x8000033C) // NOP
        return;

    uint8_t opHi = (instr >> 25) & 0x7F;
    const uint32_t pcMask = microAddressMask();

    // The lower instruction encoding uses bits 31:25 for the primary opcode
    switch (opHi)
    {
    case 0x00: // LQ (Load Quadword from VU data memory)
    {
        uint8_t it = FT(instr);  // VF destination
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            float tmp[4];
            std::memcpy(tmp, vuData + addr, 16);
            applyDest(m_state.vf[it], tmp, dest);
        }
        return;
    }
    case 0x01: // SQ (Store Quadword to VU data memory)
    {
        uint8_t is = FS(instr);  // VF source
        uint8_t it = VIT(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[it] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            uint32_t words[4]{};
            std::memcpy(words, m_state.vf[is], sizeof(words));
            queueStore(addr, words, dest);
        }
        return;
    }
    case 0x04: // ILW (Integer Load Word from VU data memory)
    {
        uint8_t it = VIT(instr); // VI destination
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            int comp = 0;
            if (dest & 0x8)
                comp = 0;
            else if (dest & 0x4)
                comp = 1;
            else if (dest & 0x2)
                comp = 2;
            else
                comp = 3;
            uint32_t v;
            std::memcpy(&v, vuData + addr + comp * 4, 4);
            if (it != 0)
                m_state.vi[it] = (int32_t)(int16_t)(v & 0xFFFF);
        }
        return;
    }
    case 0x05: // ISW (Integer Store Word to VU data memory)
    {
        uint8_t it = VIT(instr); // VI source
        uint8_t is = VIS(instr); // VI base
        uint8_t dest = (instr >> 21) & 0xF;
        int16_t imm = IMM11(instr);
        uint32_t addr = ((uint32_t)(int32_t)(m_state.vi[is] + imm)) * 16u;
        addr &= (dataSize - 1);
        if (addr + 16 <= dataSize)
        {
            const uint32_t val = static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[it] & 0xFFFF));
            const uint32_t words[4] = {val, val, val, val};
            queueStore(addr, words, dest);
        }
        return;
    }
    case 0x08: // IADDIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] + imm);
        return;
    }
    case 0x09: // ISUBIU
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = (int16_t)(instr & 0x7FF) | ((instr >> 10) & 0x7800);
        if (it != 0)
            m_state.vi[it] = (int16_t)(m_state.vi[is] - imm);
        return;
    }
    case 0x10: // FCEQ
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & 0xFFFFFF) == imm24) ? 1 : 0;
        return;
    }
    case 0x11: // FCSET
    {
        queueFcset(instr & 0xFFFFFFu);
        return;
    }
    case 0x12: // FCAND
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip & imm24) != 0) ? 1 : 0;
        return;
    }
    case 0x13: // FCOR
    {
        uint32_t imm24 = instr & 0xFFFFFF;
        if (1 != 0)
            m_state.vi[1] = ((m_state.clip | imm24) == 0xFFFFFF) ? 1 : 0;
        return;
    }
    case 0x14: // FSEQ
    {
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = ((m_state.status & 0xFFFu) == imm12) ? 1 : 0;
        return;
    }
    case 0x15: // FSSET
    {
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        queueFsset(imm12);
        return;
    }
    case 0x16: // FSAND
    {
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>((m_state.status & 0xFFFu) & imm12);
        return;
    }
    case 0x17: // FSOR
    {
        const uint8_t it = VIT(instr);
        const uint16_t imm12 = static_cast<uint16_t>((((instr >> 21) & 0x1u) << 11) | (instr & 0x7FFu));
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>((m_state.status & 0xFFFu) | imm12);
        return;
    }
    case 0x18: // FMEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = ((m_state.mac & 0xFFFF) == (uint32_t)(uint16_t)m_state.vi[is]) ? 1 : 0;
        return;
    }
    case 0x1A: // FMAND
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac & (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1B: // FMOR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        if (it != 0)
            m_state.vi[it] = (int32_t)(m_state.mac | (uint32_t)(uint16_t)m_state.vi[is]);
        return;
    }
    case 0x1C: // FCGET
    {
        const uint8_t it = VIT(instr);
        if (it != 0)
            m_state.vi[it] = static_cast<int32_t>(m_state.clip & 0x0FFFu);
        return;
    }
    case 0x20: // B (unconditional branch)
    {
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x21: // BAL (Branch and link)
    {
        uint8_t it = VIT(instr);
        int16_t imm = IMM11(instr);
        uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x24: // JR
    {
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)readBranchVi(is) * 8u) & pcMask;
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x25: // JALR
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        uint32_t target = ((uint32_t)(uint16_t)readBranchVi(is) * 8u) & pcMask;
        if (it != 0)
            m_state.vi[it] = (int32_t)((m_state.pc + 16) / 8);
        m_state.branchPending = true;
        m_state.branchTarget = target;
        m_state.branchDelay = 1;
        return;
    }
    case 0x28: // IBEQ
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) == (int16_t)readBranchVi(it))
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x29: // IBNE
    {
        uint8_t it = VIT(instr);
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) != (int16_t)readBranchVi(it))
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2C: // IBLTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) < 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2D: // IBGTZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) > 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2E: // IBLEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) <= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }
    case 0x2F: // IBGEZ
    {
        uint8_t is = VIS(instr);
        int16_t imm = IMM11(instr);
        if ((int16_t)readBranchVi(is) >= 0)
        {
            uint32_t target = (m_state.pc + 8 + imm * 8) & pcMask;
            m_state.branchPending = true;
            m_state.branchTarget = target;
            m_state.branchDelay = 1;
        }
        return;
    }

    case 0x40: // Lower1 / lower special. Bit31 set; low 6 bits select integer or special op.
    {
        const uint8_t funct = instr & 0x3Fu;
        const uint8_t vfT = FT(instr);
        const uint8_t vfS = FS(instr);
        const uint8_t viT = VIT(instr);
        const uint8_t viS = VIS(instr);
        const uint8_t viD = VID(instr);
        const uint8_t dest = (instr >> 21) & 0xF;

        switch (funct)
        {
        case 0x30: // IADD
            if (viD != 0)
                m_state.vi[viD] = (int16_t)(m_state.vi[viS] + m_state.vi[viT]);
            return;
        case 0x31: // ISUB
            if (viD != 0)
                m_state.vi[viD] = (int16_t)(m_state.vi[viS] - m_state.vi[viT]);
            return;
        case 0x32: // IADDI
        {
            int16_t imm5 = (int16_t)((int32_t)((instr >> 6) & 0x1F) << 27 >> 27);
            if (viT != 0)
                m_state.vi[viT] = (int16_t)(m_state.vi[viS] + imm5);
            return;
        }
        case 0x34: // IAND
            if (viD != 0)
                m_state.vi[viD] = m_state.vi[viS] & m_state.vi[viT];
            return;
        case 0x35: // IOR
            if (viD != 0)
                m_state.vi[viD] = m_state.vi[viS] | m_state.vi[viT];
            return;

        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F: // Lower1 special. Dobie decodes this as (instr & 3) | ((instr >> 4) & 0x7C).
        {
            const uint8_t funct2 = (uint8_t)((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
            switch (funct2)
            {
            case 0x30: // MOVE
            {
                float tmp[4];
                std::memcpy(tmp, m_state.vf[vfS], 16);
                applyDest(m_state.vf[vfT], tmp, dest);
                return;
            }
            case 0x31: // MR32 (rotate right by 32 bits = shift xyzw -> yzwx)
            {
                float tmp[4] = {m_state.vf[vfS][1], m_state.vf[vfS][2], m_state.vf[vfS][3], m_state.vf[vfS][0]};
                applyDest(m_state.vf[vfT], tmp, dest);
                return;
            }
            case 0x34: // LQI (Load Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[vfT], tmp, dest);
                }
                if (viS != 0)
                    m_state.vi[viS] = (int16_t)(m_state.vi[viS] + 1);
                return;
            }
            case 0x35: // SQI (Store Quadword, post-increment)
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, m_state.vf[vfS], sizeof(words));
                    queueStore(addr, words, dest);
                }
                if (viT != 0)
                    m_state.vi[viT] = (int16_t)(m_state.vi[viT] + 1);
                return;
            }
            case 0x36: // LQD (Load Quadword, pre-decrement)
            {
                if (viS != 0)
                    m_state.vi[viS] = (int16_t)(m_state.vi[viS] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDest(m_state.vf[vfT], tmp, dest);
                }
                return;
            }
            case 0x37: // SQD (Store Quadword, pre-decrement)
            {
                if (viT != 0)
                    m_state.vi[viT] = (int16_t)(m_state.vi[viT] - 1);
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viT]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, m_state.vf[vfS], sizeof(words));
                    queueStore(addr, words, dest);
                }
                return;
            }
            case 0x38: // DIV
            case 0x39: // SQRT
            case 0x3A: // RSQRT
            {
                float result = 0.0f;
                uint32_t statusDi = 0u, latency = 0u;
                fdivResult(instr, result, statusDi, latency);
                queueQ(result, latency, statusDi);
                return;
            }
            case 0x3B: // WAITQ
                return;
            case 0x3C: // MTIR (Move To Integer Register)
            {
                // MTIR encodes a two-bit fsf component selector in bits
                // 22:21. It is not a four-bit destination mask.
                const uint32_t comp = (instr >> 21) & 0x3u;
                uint32_t fval;
                std::memcpy(&fval, &m_state.vf[vfS][comp], 4);
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(int16_t)(fval & 0xFFFF);
                return;
            }
            case 0x3D: // MFIR (Move From Integer Register)
            {
                float result[4];
                int32_t val = (int32_t)(int16_t)(m_state.vi[viS] & 0xFFFF);
                std::memcpy(&result[0], &val, 4);
                result[1] = result[0];
                result[2] = result[0];
                result[3] = result[0];
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x3E: // ILWR - integer load word from address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    int comp = 0;
                    if (dest & 0x8)
                        comp = 0;
                    else if (dest & 0x4)
                        comp = 1;
                    else if (dest & 0x2)
                        comp = 2;
                    else
                        comp = 3;
                    uint32_t v;
                    std::memcpy(&v, vuData + addr + comp * 4, 4);
                    if (viT != 0)
                        m_state.vi[viT] = (int32_t)(int16_t)(v & 0xFFFF);
                }
                return;
            }
            case 0x3F: // ISWR - integer store word to address in VI[is]
            {
                uint32_t addr = ((uint32_t)(uint16_t)m_state.vi[viS]) * 16u;
                addr &= (dataSize - 1);
                if (addr + 16 <= dataSize)
                {
                    const uint32_t val =
                        static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viT] & 0xFFFF));
                    const uint32_t words[4] = {val, val, val, val};
                    queueStore(addr, words, dest);
                }
                return;
            }
            case 0x40: // RNEXT
            {
                const uint32_t x = (m_state.r >> 4) & 1u;
                const uint32_t y = (m_state.r >> 22) & 1u;
                m_state.r = ((m_state.r << 1) ^ x ^ y) & 0x007FFFFFu;
                m_state.r |= 0x3F800000u;
                float value = 0.0f;
                std::memcpy(&value, &m_state.r, sizeof(value));
                const float result[4] = {value, value, value, value};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x41: // RGET
            {
                float value = 0.0f;
                std::memcpy(&value, &m_state.r, sizeof(value));
                const float result[4] = {value, value, value, value};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x42: // RINIT
            {
                const uint32_t component = (instr >> 21) & 3u;
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[vfS][component], sizeof(bits));
                m_state.r = 0x3F800000u | (bits & 0x007FFFFFu);
                return;
            }
            case 0x43: // RXOR
            {
                const uint32_t component = (instr >> 21) & 3u;
                uint32_t bits = 0u;
                std::memcpy(&bits, &m_state.vf[vfS][component], sizeof(bits));
                m_state.r = 0x3F800000u | ((m_state.r ^ bits) & 0x007FFFFFu);
                return;
            }
            case 0x64: // MFP (Move From P register)
            {
                float result[4] = {m_state.p, m_state.p, m_state.p, m_state.p};
                applyDest(m_state.vf[vfT], result, dest);
                return;
            }
            case 0x68: // XTOP - move current VIF1 TOP into VI register
            {
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(m_state.top & 0x3FFu);
                return;
            }
            case 0x69: // XITOP - move current VIF1 ITOP into VI register
            {
                if (viT != 0)
                    m_state.vi[viT] = (int32_t)(m_state.itop & 0x3FFu);
                return;
            }
            case 0x6C: // XGKICK - send GIF packet from VU1 data memory
                startXgkick(static_cast<uint32_t>(static_cast<uint16_t>(m_state.vi[viS])));
                return;
            case 0x70: // ESADD
            case 0x71: // ERSADD
            case 0x72: // ELENG
            case 0x73: // ERLENG
            case 0x74: // EATANxy
            case 0x75: // EATANxz
            case 0x76: // ESUM
            case 0x77: // ERSQRT
            case 0x78: // ESQRT
            case 0x79: // ESIN
            case 0x7A: // ERCPR
            case 0x7C: // EATAN
            case 0x7D: // EEXP
            {
                float result = 0.0f;
                uint32_t latency = 0u;
                efuResult(instr, result, latency);
                queueP(result, latency);
                return;
            }
            case 0x7B: // WAITP
                return;
            default:
                reportReservedInstruction(false, instr);
                return;
            }
        }
        default:
            reportReservedInstruction(false, instr);
            return;
        }
    }
    default:
        reportReservedInstruction(false, instr);
        break;
    }
}

// One instruction pair of VU1Interpreter::run(), after decode and stall resolution: execute,
// queue the delayed writes, advance PC/branch state, handle E/D/T, then advance one cycle.
// Shared verbatim by the interpreter and by recompiled microcode (Vu1Native).
inline void VU1Interpreter::markPairWrites(const DecodedInstructionPair &decoded)
{
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    if (lowerWrite.reg != 0u &&
        decoded.suppressedLowerVf != lowerWrite.reg)
    {
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((lowerWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[lowerWrite.reg][component] = m_cycle + latency;
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    if (upperWrite.reg != 0u)
    {
        const uint32_t latency = decoded.upperUsage.vfLatency != 0u
                                     ? decoded.upperUsage.vfLatency
                                     : decoded.upperUsage.latency;
        for (uint32_t component = 0; component < 4u; ++component)
        {
            if ((upperWrite.lanes & laneForComponent(component)) != 0u)
                m_vfReady[upperWrite.reg][component] = m_cycle + latency;
        }
    }

    for (uint32_t reg = 1; reg < m_viReady.size(); ++reg)
    {
        if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
            m_viReady[reg] = m_cycle + (decoded.lowerUsage.viLatency != 0u ? decoded.lowerUsage.viLatency : decoded.lowerUsage.latency);
    }
    for (uint32_t component = 0; component < 4u; ++component)
    {
        if ((decoded.upperUsage.accWrite & laneForComponent(component)) != 0u)
            m_accReady[component] = m_cycle + kAccForwardLatency;
    }
}

template <bool Immediate, bool Flags, bool Fast, bool Plain>
inline bool VU1Interpreter::executePair(const DecodedInstructionPair &decoded, uint8_t *vuData, uint32_t dataSize,
                                        GS &gs, PS2Memory *memory, uint32_t codeSize)
{
    bool programEnded = false;
    if (Immediate && !decoded.iBit)
    {
        // FCEQ FCAND FCOR FSEQ FSAND FSOR FMEQ FMAND FMOR FCGET observe MAC/status/clip.
        const uint32_t op = (decoded.lower >> 25) & 0x7Fu;
        if (op == 0x10u || op == 0x12u || op == 0x13u || op == 0x14u || op == 0x16u || op == 0x17u ||
            op == 0x18u || op == 0x1Au || op == 0x1Bu || op == 0x1Cu)
            commitDueFlags();
    }
    uint8_t writtenVi = 0u;
    int32_t oldVi = 0;
    for (uint32_t reg = 1; reg < 16u; ++reg)
    {
        if ((decoded.lowerUsage.viWrite & (1u << reg)) != 0u)
        {
            writtenVi = static_cast<uint8_t>(reg);
            oldVi = m_state.vi[reg];
            break;
        }
    }

    const VfAccess upperWrite = decoded.upperUsage.vfWrite;
    const VfAccess lowerWrite = decoded.lowerUsage.vfWrite;
    const bool hasUpperWrite = upperWrite.reg != 0u;
    const bool hasLowerWrite = lowerWrite.reg != 0u && decoded.suppressedLowerVf != lowerWrite.reg;
    const bool hasDistinctLowerWrite = hasLowerWrite && (!hasUpperWrite || lowerWrite.reg != upperWrite.reg);
    float oldUpperVf[4]{};
    float newUpperVf[4]{};
    float oldLowerVf[4]{};
    float newLowerVf[4]{};
    float oldAcc[4]{};
    float newAcc[4]{};
    if (hasUpperWrite)
        std::memcpy(oldUpperVf, m_state.vf[upperWrite.reg], sizeof(oldUpperVf));
    if (hasDistinctLowerWrite)
        std::memcpy(oldLowerVf, m_state.vf[lowerWrite.reg], sizeof(oldLowerVf));
    if (decoded.upperUsage.accWrite != 0u)
        std::memcpy(oldAcc, m_state.acc, sizeof(oldAcc));

    if (decoded.iBit)
    {
        [[clang::always_inline]] execUpper<Flags>(decoded.upper);
        float immediate = 0.0f;
        std::memcpy(&immediate, &decoded.lower, sizeof(immediate));
        m_state.i = normalizeOperand(immediate);
    }
    else if (decoded.upperVfShadowReg != 0u)
    {
        float oldVf[4]{};
        float upperVf[4]{};
        std::memcpy(oldVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(oldVf));
        [[clang::always_inline]] execUpper<Flags>(decoded.upper);
        std::memcpy(upperVf,
                    m_state.vf[decoded.upperVfShadowReg],
                    sizeof(upperVf));
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    oldVf,
                    sizeof(oldVf));
        [[clang::always_inline]] execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
        std::memcpy(m_state.vf[decoded.upperVfShadowReg],
                    upperVf,
                    sizeof(upperVf));
    }
    else
    {
        bool done = false;
        if constexpr (Fast)
            done = execUpperFast<Flags>(decoded.upper);
        if (!done)
            [[clang::always_inline]] execUpper<Flags>(decoded.upper);
        [[clang::always_inline]] execLower(decoded.lower, vuData, dataSize, gs, memory, decoded.upper);
    }

    m_viBranchBackupValid = false;

    // Immediate (recompiled code): VF/VI/ACC reads are hazard-interlocked -- every reader stalls
    // until the newest write's ready cycle -- so writing results in program order is observably
    // identical to the delayed-write pipelines. Ready times are still tracked (markPairWrites) so
    // stalls and an interpreter continuation after a deopt behave the same.
    if (Immediate)
    {
        if constexpr (!Fast) // fast mode: ready cycles only matter to a deopt, which may stall less
            [[clang::always_inline]] markPairWrites(decoded);
        if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
            recordViWriteForBranch(writtenVi, oldVi);
    }
    else
    {
    if (hasUpperWrite)
    {
        std::memcpy(newUpperVf, m_state.vf[upperWrite.reg], sizeof(newUpperVf));
        std::memcpy(m_state.vf[upperWrite.reg], oldUpperVf, sizeof(oldUpperVf));
        const uint32_t latency =
            decoded.upperUsage.vfLatency != 0u
                ? decoded.upperUsage.vfLatency
                : decoded.upperUsage.latency;
        queueVfWrite(upperWrite.reg, upperWrite.lanes, newUpperVf, latency);
    }
    if (hasDistinctLowerWrite)
    {
        std::memcpy(newLowerVf, m_state.vf[lowerWrite.reg], sizeof(newLowerVf));
        std::memcpy(m_state.vf[lowerWrite.reg], oldLowerVf, sizeof(oldLowerVf));
        const uint32_t latency = decoded.lowerUsage.vfLatency != 0u
                                     ? decoded.lowerUsage.vfLatency
                                     : decoded.lowerUsage.latency;
        queueVfWrite(lowerWrite.reg, lowerWrite.lanes, newLowerVf, latency);
    }
    if (decoded.upperUsage.accWrite != 0u)
    {
        std::memcpy(newAcc, m_state.acc, sizeof(newAcc));
        std::memcpy(m_state.acc, oldAcc, sizeof(oldAcc));
        // ACC is forwarded to the next upper instruction. Its arithmetic
        // flags still use the normal four-cycle FMAC timeline.
        queueAccWrite(decoded.upperUsage.accWrite, newAcc,
                      kAccForwardLatency);
    }
    if (writtenVi != 0u)
    {
        const int32_t newVi = m_state.vi[writtenVi];
        m_state.vi[writtenVi] = oldVi;
        const uint32_t latency =
            decoded.lowerUsage.viLatency != 0u
                ? decoded.lowerUsage.viLatency
                : decoded.lowerUsage.latency;
        queueViWrite(writtenVi, newVi, latency);
    }

    [[clang::always_inline]] markPairWrites(decoded);
    if (writtenVi != 0u && decoded.lowerUsage.delaysNextBranchRead)
        recordViWriteForBranch(writtenVi, oldVi);
    }

    m_state.vf[0][0] = 0.0f;
    m_state.vf[0][1] = 0.0f;
    m_state.vf[0][2] = 0.0f;
    m_state.vf[0][3] = 1.0f;
    m_state.vi[0] = 0;

    if constexpr (Plain)
    {
        // Recompiled straight-line pair (no branch, delay slot, E/D/T bit): the generated code
        // continues at the next node itself and keeps m_state.pc for deopts.
        ++m_cycle;
        if (m_cycle >= m_nextCommitCycle)
            commitReadyPipelines();
        if (m_xgkick.active)
            progressXgkick();
        return false;
    }

    uint32_t nextPc = m_state.pc + 8u;
    if (nextPc >= codeSize)
        nextPc = 0u;
    m_state.pc = nextPc;

    if (m_state.branchPending)
    {
        if (m_state.branchDelay == 0u)
        {
            m_state.pc = m_state.branchTarget & microAddressMask();
            m_state.branchPending = false;
        }
        else
        {
            --m_state.branchDelay;
        }
    }

    const bool dHalt = decoded.dBit && m_state.dBitEnabled;
    const bool tHalt = decoded.tBit && m_state.tBitEnabled;
    const bool haltBit = dHalt || tHalt;
    const bool haltBranch = haltBit && decoded.lowerUsage.pipeline == PipelineBranch;

    if (m_state.haltAfterDelaySlot)
    {
        m_state.stoppedByD = m_pendingHaltD;
        m_state.stoppedByT = m_pendingHaltT;
        programEnded = true;
    }
    else if (m_state.ebit)
        programEnded = true;
    else if (haltBit && !haltBranch)
    {
        m_state.stoppedByD = dHalt;
        m_state.stoppedByT = tHalt;
        programEnded = true;
    }
    else if (decoded.eBit)
        m_state.ebit = true;
    else if (haltBranch)
    {
        m_state.haltAfterDelaySlot = true;
        m_pendingHaltD = dHalt;
        m_pendingHaltT = tHalt;
    }


    if constexpr (Fast)
    {
        ++m_cycle; // m_state.cycles is written back by endRun()
        if (m_cycle >= m_nextCommitCycle)
            commitReadyPipelines();
        if (m_xgkick.active)
            progressXgkick();
    }
    else
        advanceOneCycle();
    return programEnded;
}
