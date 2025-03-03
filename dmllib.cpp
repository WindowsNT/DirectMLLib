#include "pch.h"
#include "dmllib.hpp"


DML_BINDING_DESC MLBUFFER::BindingDesc()
{
	if (!b.b)
		throw;
	DML_BINDING_DESC dbd = {};
	dmb.Buffer = b.b;
	dmb.Offset = 0;
	dmb.SizeInBytes = b.sz();
	dbd.Type = DML_BINDING_TYPE_BUFFER;
	dbd.Desc = &dmb;
	return dbd;
}







HRESULT MLBUFFER::Create2(ID3D12Device* d3D12Device, size_t x, [[maybe_unused]] bool ForceInternal)
{
	b.b = 0;
//	b.ls = x;
	auto x1 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	auto x2 = CD3DX12_RESOURCE_DESC::Buffer(x, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	[[maybe_unused]] auto hr = d3D12Device->CreateCommittedResource(
		&x1,
		D3D12_HEAP_FLAG_NONE,
		&x2,
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&b.b));
	return hr;
}


UINT64 MLBUFFER::Upload(ML* ml,void* data, size_t by)
{
	if (!b.b)
		return 0;
	std::vector<char> bigger_data;

	if (by != b.sz())
	{
		if (by > b.sz())
		{
			throw;
		}
		else
		{
			bigger_data.resize(b.sz());
			memcpy(bigger_data.data(), data, by);
			data = bigger_data.data();
			by = b.sz();
		}
	}


	auto x1 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	auto x2 = CD3DX12_RESOURCE_DESC::Buffer(b.sz(), D3D12_RESOURCE_FLAG_NONE);
	if (!uploadBuffer)
	{
		[[maybe_unused]] auto hr = ml->d3D12Device->CreateCommittedResource(
			&x1,
			D3D12_HEAP_FLAG_NONE,
			&x2,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&uploadBuffer));
	}

	// Transition the destination buffer to COPY_DEST
	auto transitionToCopyDest = CD3DX12_RESOURCE_BARRIER::Transition(
		b.b,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, // Or current state
		D3D12_RESOURCE_STATE_COPY_DEST);
	ml->commandList->ResourceBarrier(1, &transitionToCopyDest);

	D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
	tensorSubresourceData.pData = data;
	tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(by);
	tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

	// Upload to the GPU.
	auto rv = ::UpdateSubresources(ml->commandList, b.b, uploadBuffer, 0, 0, 1, &tensorSubresourceData);
	return rv;
}


void MLOP::tape()
{
	auto bp = dmlCompiledOperator->GetBindingProperties();
	if (bp.TemporaryResourceSize)
	{
		tre.Create2(ml->d3D12Device, bp.TemporaryResourceSize, true);
		auto bu = tre.BindingDesc();
		dmlBindingTable->BindTemporaryResource(&bu);

	}
	if (bp.PersistentResourceSize)
	{
		pre.Create2(ml->d3D12Device, bp.PersistentResourceSize, true);
		auto bu = pre.BindingDesc();
		dmlBindingTable->BindPersistentResource(&bu);
	}
}


HRESULT MLOP::CreateInitializer(IDMLDevice* dmlDevice)
{
	std::vector<IDMLCompiledOperator*> dmlCompiledOperators2;
	dmlCompiledOperators2.push_back(dmlCompiledOperator);
	return dmlDevice->CreateOperatorInitializer((UINT)dmlCompiledOperators2.size(), dmlCompiledOperators2.data(), IID_PPV_ARGS(&dmlOperatorInitializer));
}



bool MLOP::ResetToExecute()
{
	dmlBindingTableDesc.Dispatchable = dmlCompiledOperator;
	return SUCCEEDED((dmlBindingTable->Reset(&dmlBindingTableDesc)));
}


UINT MLOP::FindDescriptorCount()
{
	auto  initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
	auto executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
	descriptorCount = 0;
	descriptorCount = std::max(
		initializeBindingProperties.RequiredDescriptorCount,
		std::max(descriptorCount, executeBindingProperties.RequiredDescriptorCount));
	return descriptorCount;
}

bool MLOP::CreateBindingTable(IDMLDevice* dmlDevice, ID3D12DescriptorHeap* descriptorHeap)
{
	dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer;
	dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
	dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
	dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

	return SUCCEEDED(dmlDevice->CreateBindingTable(
		&dmlBindingTableDesc,
		IID_PPV_ARGS(&dmlBindingTable)));

}



void MLOP::TransitionBindings(ID3D12GraphicsCommandList* commandList)
{
	// Transition of the buffers
	for (auto& it : items)
	{
		if (it.BindingMode == BINDING_MODE::NONE)
			continue;
		auto buffd = it.operator DML_BINDING_DESC();
		auto buff = ((DML_BUFFER_BINDING*)buffd.Desc)->Buffer;
		if (!buff)
			continue;
		auto x = CD3DX12_RESOURCE_BARRIER::Transition(buff, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
		commandList->ResourceBarrier(1, &x);
	}

	if (items.empty())
	{
		// Transition of the buffers
		for (auto& bu : bindings_in)
		{
			auto buff = ((DML_BUFFER_BINDING*)bu.Desc)->Buffer;

			auto x = CD3DX12_RESOURCE_BARRIER::Transition(buff, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			commandList->ResourceBarrier(1, &x);
		}
		for (auto& bu : bindings_out)
		{
			auto buff = ((DML_BUFFER_BINDING*)bu.Desc)->Buffer;
			auto x = CD3DX12_RESOURCE_BARRIER::Transition(buff, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
			commandList->ResourceBarrier(1, &x);
		}

	}
}

void MLOP::Bind()
{
	std::vector< DML_BINDING_DESC> bin;
	std::vector< DML_BINDING_DESC> bout;
	for (auto& it : items)
	{
		if (it.BindingMode == BINDING_MODE::BIND_IN)
		{
			auto buffd = it.operator DML_BINDING_DESC();
			bin.push_back(buffd);
		}
		if (it.BindingMode == BINDING_MODE::BIND_OUT)
		{
			auto buffd = it.operator DML_BINDING_DESC();
			bout.push_back(buffd);
		}
	}
	if (bin.size() == 0 && bout.size() == 0)
	{
		bin = bindings_in;
		bout = bindings_out;
	}


	dmlBindingTable->BindInputs((UINT)bin.size(), bin.data());
	dmlBindingTable->BindOutputs((UINT)bout.size(), bout.data());
}

void MLOP::tapi()
{
	auto bp = dmlOperatorInitializer->GetBindingProperties();
	if (bp.TemporaryResourceSize)
	{
		tri.Create2(ml->d3D12Device, bp.TemporaryResourceSize, true);
		auto bu = tri.BindingDesc();
		dmlBindingTable->BindTemporaryResource(&bu);

	}
	if (bp.PersistentResourceSize)
	{
		pri.Create2(ml->d3D12Device, bp.PersistentResourceSize, true);
		auto bu = pri.BindingDesc();
		dmlBindingTable->BindPersistentResource(&bu);
	}

}


HRESULT MLBUFFER::Download(ML* ml, size_t j, std::vector<char>& out)
{
	out.resize(0);
	CComPtr< ID3D12Resource> outputBuffer = b.b;
	if (!outputBuffer || !ml)
		return E_NOINTERFACE;

	if (j == (size_t)-1)
		j = b.sz();
	CComPtr<ID3D12Resource> readbackBuffer;
	auto x7 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
	auto x8 = CD3DX12_RESOURCE_DESC::Buffer(j);
	auto hr = ml->d3D12Device->CreateCommittedResource(
		&x7,
		D3D12_HEAP_FLAG_NONE,
		&x8,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&readbackBuffer));
	if (FAILED(hr))
		return hr;


	auto x10 = CD3DX12_RESOURCE_BARRIER::Transition(outputBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	ml->commandList->ResourceBarrier(1, &x10);
	ml->commandList->CopyResource(readbackBuffer, outputBuffer);
	if (ml)
		ml->CloseExecuteResetWait();

	D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(j) };
	void* outputBufferData = 0;
	hr = readbackBuffer->Map(0, &tensorBufferRange, &outputBufferData);
	if (FAILED(hr))
		return hr;
	out.resize(j);
	memcpy(out.data(), ((char*)outputBufferData + 0), out.size());
	return S_OK;

}

void ML::SetDescriptorHeaps()
{
	ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { descriptorHeap };
	commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);
}


void  ML::Record(int what)
{
	if (what == 0)
	{
		for (auto& op : ops)
			dmlCommandRecorder->RecordDispatch(commandList, op.dmlOperatorInitializer, op.dmlBindingTable);
	}
	if (what == 1)
	{
		for (auto& op : ops)
			dmlCommandRecorder->RecordDispatch(commandList, op.dmlCompiledOperator, op.dmlBindingTable);
	}
}


void  ML::Prepare()
{
	for (auto& op : ops)
		op.CreateInitializer(dmlDevice);

	// https://learn.microsoft.com/en-us/windows/ai/directml/dml-binding
	// Query the operator for the required size (in descriptors) of its binding table.
	CreateHeap();

	// Create a binding table over the descriptor heap we just created.
	for (auto& op : ops)
		op.CreateBindingTable(dmlDevice, descriptorHeap);

	// Bind Temporary and 
	for (auto& op : ops)
		op.tapi();

	// The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
	CreateCommandRecorder();

	// Record execution of the operator initializer.
	Record(0);

	// Execute it
	CloseExecuteResetWait();
}

void ML::Run(size_t which)
{
	SetDescriptorHeaps();

	// Run it
	for(size_t i = 0 ; i < ops.size() ; i++)
	{
		if (which != (size_t)-1 && i != which)
			continue;
		auto& op = ops[i];
		op.ResetToExecute();
		op.TransitionBindings(commandList);

		// Binding
		if (1)
		{
			op.Bind();
		}
		else
		{
//			op.dmlBindingTable->BindInputs((UINT)op.bindings_in.size(), op.bindings_in.data());
//			op.dmlBindingTable->BindOutputs((UINT)op.bindings_out.size(), op.bindings_out.data());
		}

		// And temporary/persistent resources
		op.tape();

		// And run it
		dmlCommandRecorder->RecordDispatch(commandList, op.dmlCompiledOperator, op.dmlBindingTable);
		CloseExecuteResetWait();
		SetDescriptorHeaps();
	}

}

bool ML::CreateHeap()
{
	// You need to initialize an operator exactly once before it can be executed, and
	// the two stages require different numbers of descriptors for binding. For simplicity,
	// we create a single descriptor heap that's large enough to satisfy them both.
	UINT descriptorCount = 0;
	for (auto& op : ops)
		descriptorCount += op.FindDescriptorCount();

	if (descriptorCount == 0)
		descriptorCount = 1;
	// Create descriptor heaps.

	D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
	descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
	descriptorHeapDesc.NumDescriptors = descriptorCount;
	descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
	descriptorHeap = 0;
	if (FAILED((d3D12Device->CreateDescriptorHeap(
		&descriptorHeapDesc,
		IID_PPV_ARGS(&descriptorHeap)))))
		return false;

	// Set the descriptor heap(s).
	SetDescriptorHeaps();
	return true;
}



bool ML::CloseExecuteResetWait()
{
	if (FAILED((commandList->Close())))
		return false;

	ID3D12CommandList* commandLists[] = { commandList };
	commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

	CComPtr<ID3D12Fence> d3D12Fence;
	if (FAILED((d3D12Device->CreateFence(
		0,
		D3D12_FENCE_FLAG_NONE,
		IID_PPV_ARGS(&d3D12Fence)))))
		return false;

	auto hfenceEventHandle = ::CreateEvent(nullptr, true, false, nullptr);

	if (FAILED(commandQueue->Signal(d3D12Fence, 1)))
		return false;
	if (FAILED((d3D12Fence->SetEventOnCompletion(1, hfenceEventHandle))))
		return false;

	WaitForSingleObjectEx(hfenceEventHandle, INFINITE, FALSE);

	if (FAILED((commandAllocator->Reset())))
		return false;
	if (FAILED((commandList->Reset(commandAllocator, nullptr))))
		return false;
	CloseHandle(hfenceEventHandle);
	return true;
}


dml::Expression ML::ConstantValueTensor(dml::Graph& graph, float what, dml::TensorDesc::Dimensions outputSizes)
{
	DML_SCALAR_UNION scalar2;
	scalar2.Float32 = what;
	auto zT = dml::FillValueConstant(
		graph, outputSizes,
		DML_TENSOR_DATA_TYPE_FLOAT32,       // Data type
		scalar2
	);
	return zT;
}


HRESULT ML::CreateCommandRecorder()
{
	dmlCommandRecorder = 0;
	return    dmlDevice->CreateCommandRecorder(
		IID_PPV_ARGS(&dmlCommandRecorder));
}


HRESULT ML::CreateDML()
{
	if (dmlDevice)
		return S_FALSE;
	DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
#ifdef _DEBUG
	if (Debug)
		dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
	DMLCreateDevice1(d3D12Device, dmlCreateDeviceFlags, dmlMinFeatureLevel,IID_PPV_ARGS(&dmlDevice));
	if (!dmlDevice)
		return E_FAIL;

	return S_OK;
}


ML::ML(bool dbg)
{
	Debug = dbg;
}

HRESULT ML::On()
{
	auto hr = InitializeDirect3D12();
	if (FAILED(hr))
		return hr;
	hr = CreateDML();
	if (FAILED(hr))
		return hr;

	return S_OK;
}


HRESULT ML::InitializeDirect3D12()
{
	if (d3D12Device)
		return S_FALSE;

#ifdef _DEBUG
	if (Debug)
	{
		CComPtr<ID3D12Debug> d3D12Debug;
		D3D12GetDebugInterface(IID_PPV_ARGS(&d3D12Debug));
		if (d3D12Debug)
			d3D12Debug->EnableDebugLayer();
	}
#endif

	CComPtr<IDXGIFactory4> dxgiFactory;
	CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));

	CComPtr<IDXGIAdapter> dxgiAdapter;
	UINT adapterIndex{};
	HRESULT hr{};
	do
	{
		dxgiAdapter = nullptr;
		dxgiAdapter = 0;
		if (FAILED((dxgiFactory->EnumAdapters(adapterIndex, &dxgiAdapter))))
			return E_FAIL;
		++adapterIndex;

		d3D12Device = 0;
		hr = ::D3D12CreateDevice(
			dxgiAdapter,
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&d3D12Device));
		if (hr == DXGI_ERROR_UNSUPPORTED) continue;
		if (FAILED(hr))
			return hr;
	} while (hr != S_OK);

	D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
	commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

	commandQueue = 0;
	hr = (d3D12Device->CreateCommandQueue(
		&commandQueueDesc,
		IID_PPV_ARGS(&commandQueue)));
	if (FAILED(hr))
		return hr;

	commandAllocator = 0;
	hr = (d3D12Device->CreateCommandAllocator(
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		IID_PPV_ARGS(&commandAllocator)));
	if (FAILED(hr))
		return hr;

	commandList = 0;
	hr = (d3D12Device->CreateCommandList(
		0,
		D3D12_COMMAND_LIST_TYPE_DIRECT,
		commandAllocator,
		nullptr,
		IID_PPV_ARGS(&commandList)));
	if (FAILED(hr))
		return hr;

	return S_OK;
}


MLOP::MLOP(ML* _ml)
{
	ml = _ml;
	graph = std::make_shared<dml::Graph>(ml->dmlDevice.p);
}


size_t MLOP::Count()
{
	return items.size();
}

MLOP_ITEM& MLOP::Item(size_t i)
{
	if (items.size() <= i)
		throw;
	return items[i];
}

MLOP& MLOP::AddItem(dml::Expression expr, LPARAM tag, bool NewBuffer, BINDING_MODE Binding, std::optional<MLRESOURCE> bds, uint32_t nit)
{
	if (tag != 0 && WithTag2(tag))
		throw;
	MLOP_ITEM item;
	item.tag = tag;
	item.InputTensorTag = nit;
	item.BindingMode = Binding;
	item.expr = expr;
	if (NewBuffer)
	{
		item.buffer = MLBUFFER();
		item.buffer->Create(ml->d3D12Device, expr);
	}
	item.bds = bds;
	items.push_back(item);
	return *this;

}

MLOP& MLOP::AddIntermediate(dml::Expression td, LPARAM tag)
{
	return AddItem(td, tag, 0, BINDING_MODE::NONE, {}, 0);
}


MLOP& MLOP::AddOutput(dml::Expression td, LPARAM tag)
{
	return AddItem(td, tag, true, BINDING_MODE::BIND_OUT, {}, 0);
}

MLOP& MLOP::AddInput(dml::TensorDesc td, LPARAM tag, bool NewBuffer, BINDING_MODE Binding, std::optional<MLRESOURCE> bds)
{
	uint32_t na = 0;
	for (auto& it : items)
	{
		if (it.InputTensorTag > 0)
			na = std::max(na, it.InputTensorTag);
	}
	auto expr = dml::InputTensor(*graph, na, td);
	return AddItem(expr, tag, NewBuffer, Binding, bds,na + 1);
}


MLOP_ITEM* MLOP::WithTag2(LPARAM tag)
{
	for (auto& i : items)
	{
		if (i.tag == tag)
			return &i;
	}
	return nullptr;
}

MLOP_ITEM& MLOP::WithTag(LPARAM tag)
{
	for (auto& i : items)
	{
		if (i.tag == tag)
			return i;
	}
	throw;
}


MLOP& MLOP::Build()
{
	// Inputs
	/*
	for (auto& i : items)
	{
		if (i.BindingMode == BINDING_MODE::BIND_IN)
			bindings_in.push_back(i);
	}

	// Outputs
	for (auto& i : items)
	{
		if (i.BindingMode == BINDING_MODE::BIND_OUT)
			bindings_out.push_back(i);
	}
	*/

	std::vector<dml::Expression> outputs2;
	for (auto& o : items)
	{
		if (o.BindingMode == BINDING_MODE::BIND_OUT && o.buffer)
			outputs2.push_back(o);
	}
	auto OutputCompiledOperator2 = graph->Compile(DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, outputs2);
	dmlCompiledOperator.Attach(OutputCompiledOperator2.Detach());
	return *this;
}
