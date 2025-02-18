#include "pch.h"
#include "dmllib.hpp"


DML_BINDING_DESC MLBUFFER::BindingDesc()
{
	if (!b)
		throw;
	DML_BINDING_DESC dbd = {};
	dmb.Buffer = b;
	dmb.Offset = offset;
	dmb.SizeInBytes = ls;
	dbd.Type = DML_BINDING_TYPE_BUFFER;
	dbd.Desc = &dmb;
	return dbd;
}







HRESULT MLBUFFER::Create2(ID3D12Device* d3D12Device, size_t x, [[maybe_unused]] bool ForceInternal)
{
	b = 0;
	ls = x;
	auto x1 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
	auto x2 = CD3DX12_RESOURCE_DESC::Buffer(ls, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
	[[maybe_unused]] auto hr = d3D12Device->CreateCommittedResource(
		&x1,
		D3D12_HEAP_FLAG_NONE,
		&x2,
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&b));
	return hr;
}


UINT64 MLBUFFER::Upload(ML* ml,void* data, size_t by)
{
	if (!b)
		return 0;
	std::vector<char> bigger_data;

	if (by != ls)
	{
		if (by > ls)
		{

		}
		else
		{
			bigger_data.resize(ls);
			memcpy(bigger_data.data(), data, by);
			data = bigger_data.data();
			by = ls;
		}
	}

	auto x1 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
	auto x2 = CD3DX12_RESOURCE_DESC::Buffer(ls, D3D12_RESOURCE_FLAG_NONE);
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
		b,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS, // Or current state
		D3D12_RESOURCE_STATE_COPY_DEST);
	ml->commandList->ResourceBarrier(1, &transitionToCopyDest);

	D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
	tensorSubresourceData.pData = data;
	tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(by);
	tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

	// Upload the input tensor to the GPU.
	auto rv = ::UpdateSubresources(ml->commandList, b, uploadBuffer, offset, 0, 1, &tensorSubresourceData);
	return rv;
}


void MLOP::tape(ID3D12Device* d3D12Device)
{
	auto bp = dmlCompiledOperator->GetBindingProperties();
	if (bp.TemporaryResourceSize)
	{
		tre.Create2(d3D12Device, bp.TemporaryResourceSize, true);
		auto bu = tre.BindingDesc();
		dmlBindingTable->BindTemporaryResource(&bu);

	}
	if (bp.PersistentResourceSize)
	{
		pre.Create2(d3D12Device, bp.PersistentResourceSize, true);
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


UINT MLOP::FindDC()
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


void MLOP::tapi(ID3D12Device* d3D12Device)
{
	auto bp = dmlOperatorInitializer->GetBindingProperties();
	if (bp.TemporaryResourceSize)
	{
		tri.Create2(d3D12Device, bp.TemporaryResourceSize, true);
		auto bu = tri.BindingDesc();
		dmlBindingTable->BindTemporaryResource(&bu);

	}
	if (bp.PersistentResourceSize)
	{
		pri.Create2(d3D12Device, bp.PersistentResourceSize, true);
		auto bu = pri.BindingDesc();
		dmlBindingTable->BindPersistentResource(&bu);
	}

}


HRESULT MLBUFFER::Download(ML* ml, size_t j, std::vector<char>& out)
{
	out.resize(0);
	CComPtr< ID3D12Resource> outputBuffer = b;
	if (!outputBuffer || !ml)
		return E_NOINTERFACE;

	if (j == (size_t)-1)
		j = ls;
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

	D3D12_RANGE tensorBufferRange{ offset, static_cast<SIZE_T>(j) };
	void* outputBufferData = 0;
	hr = readbackBuffer->Map(0, &tensorBufferRange, &outputBufferData);
	if (FAILED(hr))
		return hr;
	out.resize(j);
	memcpy(out.data(), ((char*)outputBufferData + offset), out.size());
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
		op.tapi(d3D12Device);

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
		op.dmlBindingTable->BindInputs((UINT)op.bindings_in.size(), op.bindings_in.data());
		op.dmlBindingTable->BindOutputs((UINT)op.bindings_out.size(), op.bindings_out.data());

		// And temporary/persistent resources
		op.tape(d3D12Device);

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
		descriptorCount += op.FindDC();

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
#if defined (_DEBUG)
#ifdef DEBUGML
	dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
#endif
	DMLCreateDevice(d3D12Device, dmlCreateDeviceFlags, IID_PPV_ARGS(&dmlDevice));
	if (!dmlDevice)
		return E_FAIL;

	return S_OK;
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

	// Throws if the D3D12 debug layer is missing - you must install the Graphics Tools optional feature
#if defined (_DEBUG)
#ifdef DEBUGML
	CComPtr<ID3D12Debug> d3D12Debug;
	D3D12GetDebugInterface(IID_PPV_ARGS(&d3D12Debug));
	if (d3D12Debug)
		d3D12Debug->EnableDebugLayer();
#endif
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


DMLOPBUILDER::DMLOPBUILDER(ID3D12Device* d3D12Device, IDMLDevice* dml) : graph(dml)
{
	this->d3D12Device = d3D12Device;
}


MLBUFFER& DMLOPBUILDER::Input(size_t i)
{
	if (inputs.size() <= i)
		throw;
	return inputs[i];
}

MLBUFFER& DMLOPBUILDER::Output(size_t i)
{
	if (outputs.size() <= i)
		throw;
	return outputs[i];
}

dml::Expression& DMLOPBUILDER::Intermediate(size_t i)
{
	if (intermediates.size() <= i)
		throw;
	return intermediates[i];
}



DMLOPBUILDER& DMLOPBUILDER::AddInput(dml::TensorDesc td)
{
	auto expr = dml::InputTensor(graph, (uint32_t)inputs.size(), td);
	MLBUFFER in;
	in.Create(d3D12Device, expr);
	inputs.emplace_back(in);
	return *this;
}


DMLOPBUILDER& DMLOPBUILDER::AddIntermediate(dml::Expression td)
{
	intermediates.push_back(td);
	return *this;
}

DMLOPBUILDER& DMLOPBUILDER::AddOutput(dml::Expression e)
{
	MLBUFFER out1;
	out1.Create(d3D12Device, e);
	outputs.push_back(out1);
	return *this;
}


DMLOPBUILDER& DMLOPBUILDER::AddToOutput(MLBUFFER& out)
{
	outputs2.push_back(out.ee);
	return *this;
}



MLOP& DMLOPBUILDER::Build()
{
	for (auto& i : inputs)
		bindings_in.push_back(i.BindingDesc());
	for (auto& i : outputs)
		bindings_out.push_back(i.BindingDesc());

	auto OutputCompiledOperator2 = graph.Compile(DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, outputs2);
	mlop.bindings_in = bindings_in;
	mlop.bindings_out = bindings_out;
	mlop.dmlCompiledOperator.Attach(OutputCompiledOperator2.Detach());
	return mlop;
}
