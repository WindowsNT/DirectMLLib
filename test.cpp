#include "pch.h"
#include "dmllib.hpp"




int main()
{
	CoInitializeEx(NULL, COINIT_MULTITHREADED);
	ML ml(true);
	auto hr = ml.On();
	if (FAILED(hr))
		return 0;


	MLOP op1(&ml);
	op1.
		AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 10,10} }).
		AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 10,10} }).
		AddIntermediate(dml::Sin(op1.Item(0))).
		AddIntermediate(dml::Cos(op1.Item(2))).
		AddOutput(dml::Add(op1.Item(3), op1.Item(1)));
	ml.ops.push_back(op1.Build());

	// Initialize
	ml.Prepare();

	// Run it 5 times
	for (int y = 0; y < 5; y++)
	{
		// Upload data	
		std::vector<float> data(100);
		for (int i = 0; i < 100; i++)
			data[i] = (float)(i * (y + 1));
		op1.Item(0).buffer->Upload(&ml, data.data(), data.size() * sizeof(float));
		op1.Item(1).buffer->Upload(&ml, data.data(), data.size() * sizeof(float));

		ml.Run();

		// Download data
		std::vector<float> fdata(100);
		std::vector<char> cdata(400);
		op1.Item(4).buffer->Download(&ml, 400, cdata);
		memcpy(fdata.data(), cdata.data(), 400);
		}

}

