#pragma once

// Save states: the GS register structs, field by field (gs_types.h).

#include "runtime/gs/gs_types.h"
#include "ps2x/state_archive.h"

namespace ps2x::gs_state
{
    inline void io(StateArchive &a, GSFrameReg &r) { a & r.fbp & r.fbw & r.psm & r.fbmsk; }
    inline void io(StateArchive &a, GSZbufReg &r) { a & r.zbp & r.psm & r.zmask; }
    inline void io(StateArchive &a, GSScissorReg &r) { a & r.x0 & r.x1 & r.y0 & r.y1; }
    inline void io(StateArchive &a, GSTex0Reg &r)
    {
        a & r.tbp0 & r.tbw & r.psm & r.tw & r.th & r.tcc & r.tfx & r.cbp & r.cpsm & r.csm & r.csa & r.cld;
    }
    inline void io(StateArchive &a, GSXYOffsetReg &r) { a & r.ofx & r.ofy; }
    inline void io(StateArchive &a, GSTexaReg &r) { a & r.ta0 & r.aem & r.ta1; }
    inline void io(StateArchive &a, GSTexClutReg &r) { a & r.cbw & r.cou & r.cov; }
    inline void io(StateArchive &a, GSPrimReg &r) { a & r.type & r.iip & r.tme & r.fge & r.abe & r.aa1 & r.fst & r.ctxt & r.fix; }
    inline void io(StateArchive &a, GSContext &c)
    {
        io(a, c.frame);
        io(a, c.scissor);
        io(a, c.tex0);
        io(a, c.xyoffset);
        io(a, c.zbuf);
        a & c.tex1 & c.miptbp1 & c.miptbp2 & c.clamp & c.alpha & c.test & c.fba;
    }
    inline void io(StateArchive &a, GSBitBltBuf &r) { a & r.sbp & r.sbw & r.spsm & r.dbp & r.dbw & r.dpsm; }
    inline void io(StateArchive &a, GSTrxPos &r) { a & r.ssax & r.ssay & r.dsax & r.dsay & r.dir; }
    inline void io(StateArchive &a, GSTrxReg &r) { a & r.rrw & r.rrh; }
    inline void io(StateArchive &a, GSVertex &v)
    {
        a & v.x & v.y & v.z & v.r & v.g & v.b & v.a & v.q & v.s & v.t & v.u & v.v & v.fog & v.motion;
    }
    inline void io(StateArchive &a, GSTransferCommand &c)
    {
        io(a, c.bitbltbuf);
        io(a, c.trxpos);
        io(a, c.trxreg);
        a & c.direction;
    }
    inline void io(StateArchive &a, GSTransferSnapshot &t)
    {
        uint64_t pending = t.localToHostPendingBytes;
        a & t.x & t.y & t.totalPixels & t.copiedPixels & t.direction & pending;
        t.localToHostPendingBytes = static_cast<size_t>(pending);
    }
}
