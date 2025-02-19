# DirectML Lib

A quick way to use Direct ML for machine learning with a clean interface.

# Quick Usage

```
int main()
{
	CoInitializeEx(NULL, COINIT_MULTITHREADED);
	ML ml(true);
	auto hr = ml.On();
	if (FAILED(hr))
		return 0;



	MLOP op1(ml.d3D12Device, ml.dmlDevice);
	op1.
		AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 10,10} }).
		AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 10,10} }).
		AddIntermediate(dml::Sin(op1.Input(0))).
		AddIntermediate(dml::Cos(op1.Intermediate(0))).
		AddOutput(dml::Add(op1.Intermediate(1), op1.Input(1))).
		AddToOutput(op1.Output(0));
	ml.ops.push_back(op1.Build());

/* Example of Gemm
	MLOP op2(ml.d3D12Device, ml.dmlDevice);
	op2.AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 1,1,10,100} });
	op2.AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 1,1,100,4} });
	dml::GemmBuilder gb(op2.Input(0),op2.Input(1));
	op2.AddOutput(gb.Build());
	op2.AddToOutput(op2.Output(0));	
	ml.ops.push_back(op2.Build());
*/

	// Initialize
	ml.Prepare();

	for (int y = 0; y < 5; y++)
	{
		// Upload data	
		std::vector<float> data(100);
		for (int i = 0; i < 100; i++)
			data[i] = (float)(i * (y + 1));
		op1.Input(0).Upload(&ml, data.data(), data.size() * sizeof(float));
		op1.Input(1).Upload(&ml, data.data(), data.size() * sizeof(float));

		ml.Run();

		// Download 
		std::vector<float> fdata(100);
		std::vector<char> cdata(400);
		op1.Output(0).Download(&ml, 400, cdata);
		memcpy(fdata.data(), cdata.data(), 400);
	}

}
```
