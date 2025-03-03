//
#pragma once

class ML;

struct MLRESOURCE
{
	CComPtr<ID3D12Resource> b = 0;
//	size_t ls = 0;

	size_t sz()
	{
		if (!b)
			return 0;
	//	return ls;
		auto de = b->GetDesc();
		return de.Width;
	}

	operator bool()
	{
		return b && sz();
	}
};

struct MLBUFFER
{
	MLRESOURCE b;
	LPARAM Tag = 0;

	static unsigned long long TensorSizeAlign(unsigned long long t)
	{
		while (t % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
			t++;
		return t;
	}

	~MLBUFFER()
	{
	}

	auto GetOutputDesc() {
		return ee.GetOutputDesc();
	}

	dml::Expression ee;


	operator dml::Expression()
	{
		return ee;
	}

	HRESULT Create(ID3D12Device* d, dml::Expression e)
	{
		ee = e;
		auto x = TensorSizeAlign(e.GetOutputDesc().totalTensorSizeInBytes);
		return Create2(d, x, false);
	}
	HRESULT Create2(ID3D12Device* d3D12Device, size_t x, [[maybe_unused]] bool ForceInternal);

	CComPtr<ID3D12Resource> uploadBuffer;
	std::vector<char> global_upload;
	UINT64 Upload(ML* ml, void* data, size_t by);
	DML_BUFFER_BINDING dmb = {};
	DML_BINDING_DESC BindingDesc();
	HRESULT Download(class ML* ml, size_t j, std::vector<char>& out);
};


enum class BINDING_MODE
{
	NONE = 0,
	BIND_IN = 1,
	BIND_OUT = 2
};

struct MLOP_ITEM
{
	LPARAM tag = 0;
	unsigned int InputTensorTag = 0; // 1 based
	BINDING_MODE BindingMode = BINDING_MODE::NONE; 
	
	std::optional<MLBUFFER> buffer;
	dml::Expression expr;
	std::optional<MLRESOURCE> bds;

	operator dml::Expression()
	{
		return expr;
	}


	// Cache
	DML_BINDING_DESC dbd = {};
	DML_BUFFER_BINDING dmb = {};
	operator DML_BINDING_DESC ()
	{
		if (buffer)
			return buffer->BindingDesc();
		if (bds)
		{
			dmb.Buffer = bds->b;
			dmb.Offset = 0;
			dmb.SizeInBytes = bds->sz();
			dbd.Type = DML_BINDING_TYPE_BUFFER;
			dbd.Desc = &dmb;
			return dbd;
		}
		throw;
	}

};

class MLOP
{
private:

	MLBUFFER tri, tre, pri, pre;
	friend class ML;
	friend class NN;

	std::shared_ptr<dml::Graph> graph;
	ML* ml = 0;


	std::vector<DML_BINDING_DESC> bindings_in;
	std::vector<DML_BINDING_DESC> bindings_out;

	CComPtr<IDMLCompiledOperator> dmlCompiledOperator;
	CComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;

	DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
	CComPtr<IDMLBindingTable> dmlBindingTable;
	UINT descriptorCount = 0;

	void tapi();
	void tape();
	HRESULT CreateInitializer(IDMLDevice* dmlDevice);

	void Bind();
	void TransitionBindings(ID3D12GraphicsCommandList* commandList);
	bool ResetToExecute();
	UINT FindDescriptorCount();
	bool CreateBindingTable(IDMLDevice* dmlDevice, ID3D12DescriptorHeap* descriptorHeap);

	std::vector<MLOP_ITEM> items;

public:



	std::shared_ptr<dml::Graph> GetGraph()
	{
		return graph;
	}
	MLOP(ML* ml);
	size_t Count();
	MLOP_ITEM& Item(size_t i);
	MLOP_ITEM& WithTag(LPARAM tag);
	MLOP_ITEM* WithTag2(LPARAM tag);

	MLOP& AddInput(dml::TensorDesc td, LPARAM tag = 0, bool NewBuffer = 1, BINDING_MODE Binding = BINDING_MODE::BIND_IN, std::optional<MLRESOURCE> bds = {});
	MLOP& AddItem(dml::Expression td, LPARAM tag = 0, bool NewBuffer = 0, BINDING_MODE Binding = BINDING_MODE::NONE, std::optional<MLRESOURCE> bds = {}, uint32_t nit = 0);
	MLOP& AddIntermediate(dml::Expression td, LPARAM tag = 0);
	MLOP& AddOutput(dml::Expression td, LPARAM tag = 0);

	MLOP& Build();

		
};

class ML
{
private:

	bool Debug = 0;
	DML_FEATURE_LEVEL dmlMinFeatureLevel = DML_FEATURE_LEVEL_1_0;

public:
	CComPtr<ID3D12Device> d3D12Device;
	CComPtr<IDMLDevice> dmlDevice;
	CComPtr<ID3D12CommandQueue> commandQueue;
	CComPtr<ID3D12CommandAllocator> commandAllocator;
	CComPtr<ID3D12GraphicsCommandList> commandList;
	CComPtr<IDMLCommandRecorder> dmlCommandRecorder;
	DML_BINDING_PROPERTIES initializeBindingProperties = {}, executeBindingProperties = {};
	CComPtr<ID3D12DescriptorHeap> descriptorHeap;

	ML(bool dbg = 0);
	void SetDebug(bool d) {
		Debug = d;
	}
	void SetFeatureLevel(DML_FEATURE_LEVEL f) {
		dmlMinFeatureLevel = f;
	}
	HRESULT On();
	HRESULT InitializeDirect3D12();
	HRESULT CreateDML();
	dml::Expression ConstantValueTensor(dml::Graph& graph, float what, dml::TensorDesc::Dimensions outputSizes);
	HRESULT CreateCommandRecorder();
	bool CloseExecuteResetWait();
	void SetDescriptorHeaps();
	bool CreateHeap();
	void Record(int what);
	void Prepare();
	void Run(size_t which = (size_t)-1);

	std::vector<MLOP> ops;

};


