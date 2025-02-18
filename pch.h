#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <vector>
#include <optional>

#include <d3d12.h>
#include "d3dx12.h"

#include <dxgi1_4.h>

#include <atlbase.h>
#include <combaseapi.h>

#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
#include "DirectMLX.h"

#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"directml.lib")

#include <windows.h>
#include <d3d11.h>
#include <directml.h>

#undef max
#undef min
