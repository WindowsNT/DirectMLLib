//


class ML;

struct MLBUFFER
{
	CComPtr<ID3D12Resource> b = 0;
	size_t ls = 0;
	size_t offset = 0;

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


struct MLOP
{
	CComPtr<IDMLCompiledOperator> dmlCompiledOperator;
	CComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;



	DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
	CComPtr<IDMLBindingTable> dmlBindingTable;
	UINT descriptorCount = 0;

	void tapi(ID3D12Device* d3D12Device);
	void tape(ID3D12Device* d3D12Device);
	HRESULT CreateInitializer(IDMLDevice* dmlDevice);

	void TransitionBindings(ID3D12GraphicsCommandList* commandList);
	bool ResetToExecute();
	UINT FindDC();
	bool CreateBindingTable(IDMLDevice* dmlDevice, ID3D12DescriptorHeap* descriptorHeap);

	MLBUFFER tri, tre, pri, pre;


	std::shared_ptr<dml::Graph> graph;
	ID3D12Device* d3D12Device = 0;
	std::vector<MLBUFFER> inputs;
	std::vector<MLBUFFER> outputs;
	std::vector<dml::Expression> intermediates;
	std::vector<DML_BINDING_DESC> bindings_in;
	std::vector<DML_BINDING_DESC> bindings_out;
	std::vector<dml::Expression> outputs2;


	MLOP(ID3D12Device* d3D12Device, IDMLDevice* dml);
	MLBUFFER& Input(size_t i);
	MLBUFFER& Output(size_t i);
	dml::Expression& Intermediate(size_t i);
	MLOP& AddInput(dml::TensorDesc td);
	MLOP& AddIntermediate(dml::Expression td);
	MLOP& AddOutput(dml::Expression e);
	MLOP& AddToOutput(MLBUFFER& out);
	MLOP& Build();


};

class ML
{
private:

	bool Debug = 0;

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


