#pragma once

// Save states: the HLE stubs' own state, one function per source file (each defined at the end of
// its file, where the file's globals are visible). Called by SaveState.cpp.

namespace ps2x
{
    class StateArchive;
}

namespace ps2_stubs
{
    void serializePadState(ps2x::StateArchive &ar);
    void serializeMemoryCardState(ps2x::StateArchive &ar);
    void serializeCdState(ps2x::StateArchive &ar);
    void serializeSifState(ps2x::StateArchive &ar);
    void serializeAudioStubState(ps2x::StateArchive &ar);
    void serializeDmaStubState(ps2x::StateArchive &ar);
    void serializeGsStubState(ps2x::StateArchive &ar);
    void serializeFileIoStubState(ps2x::StateArchive &ar);
    void serializeLibcFileState(ps2x::StateArchive &ar);
    // After a load from the app's menu: the next sceMcGetInfo per port reports a changed card,
    // so the game reads the card again (it may hold other saves than when the state was taken).
    void markMemoryCardsChanged();
}

namespace ps2_syscalls
{
    void serializeRpcState(ps2x::StateArchive &ar);
}
