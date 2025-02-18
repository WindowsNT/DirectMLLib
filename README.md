# DirectML Lib

A quick way to use Direct ML for machine learning.

# Quick Usage

```
int main()
{
	CoInitializeEx(NULL, COINIT_MULTITHREADED);
	ML ml;
	auto hr = ml.On();
	if (FAILED(hr))
		return 0;

	DMLOPBUILDER ob(ml.d3D12Device, ml.dmlDevice);
	ob.
		AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 10,10} }).
		AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 10,10} }).
		AddOutput(dml::Add(ob.Input(0), ob.Input(1))).
		AddToOutput(ob.Output(0));
	ml.ops.push_back(ob.Build());


	// Initialize
	ml.Prepare();

	for (int y = 0; y < 5; y++)
	{
		// Upload data
		std::vector<float> data(100);
		for (int i = 0; i < 100; i++)
			data[i] = (float)(i * (y + 1));
		ob.Input(0).Upload(&ml, data.data(), data.size() * sizeof(float));
		ob.Input(1).Upload(&ml, data.data(), data.size() * sizeof(float));

		ml.Run();

		// Download 
		std::vector<float> fdata(100);
		std::vector<char> cdata(400);
		ob.Output(0).Download(&ml, 400, cdata);
		memcpy(fdata.data(), cdata.data(), 400);
	}

}
```
