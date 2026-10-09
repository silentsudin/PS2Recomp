#pragma once

// Small POSIX differences the runtime hits when built for Windows (mingw-w64 has no setenv).

#include <cstdlib>

namespace ps2x
{
    // setenv(name, value, overwrite); returns 0 on success.
    inline int setEnv(const char *name, const char *value, bool overwrite = true)
    {
#ifdef _WIN32
        if (!overwrite && std::getenv(name))
            return 0;
        return _putenv_s(name, value);
#else
        return ::setenv(name, value, overwrite ? 1 : 0);
#endif
    }
}
