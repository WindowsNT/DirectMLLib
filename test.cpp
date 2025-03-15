#include "pch.h"
#include "dmllib.hpp"


auto LinearRegressionCPU(float* px, float* py, size_t n)
{
	// Sx
	float Sx = 0, Sy = 0, Sxy = 0, Sx2 = 0;
	for (size_t i = 0; i < n; i++)
	{
		Sx += px[i];
		Sx2 += px[i] * px[i];
		Sy += py[i];
		Sxy += px[i] * py[i];
	}
	float B = (n * Sxy - Sx * Sy) / ((n * Sx2) - (Sx * Sx));
	float A = (Sy - (B * Sx)) / n;

	printf("Linear Regression CPU:\r\nSx = %f\r\nSy = %f\r\nSxy = %f\r\nSx2 = %f\r\nA = %f\r\nB = %f\r\n\r\n", Sx, Sy, Sxy, Sx2, A, B);
	return std::tuple<float, float>(A, B);
}

std::vector<float> xs = { 10,15,20,25,30,35 };
std::vector<float> ys = { 1003,1005,1010,1008,1014,1022 };
size_t N = xs.size();


#include <random>
std::random_device rd;
std::mt19937 e2(rd());

void RandomData()
{
	xs.clear();
	ys.clear();

	unsigned long long how = 1024 * 1024 * 8;

	printf("Generating %zi random floats...\r\n", how);
	xs.resize(how);
	ys.resize(how);

	N = xs.size();

	std::uniform_real_distribution<> dist1(0.0f, 1.0f);
	std::uniform_real_distribution<> dist2(10.0f, 100.0f);

	for (size_t i = 0; i < N; i++)
	{
		xs[i] = (float)dist1(e2);
		ys[i] = (float)dist2(e2);
	}


}




void LinearRegressionDML(float* px, float* py, unsigned int n)
{
	ML ml(true);
	ml.SetFeatureLevel(DML_FEATURE_LEVEL_6_4);
	auto hr = ml.On();
	if (FAILED(hr))
		return;

	MLOP op1(&ml);
	
	// Input X [0]
	op1.AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 1,n} });

	// Input Y [1]
	op1.AddInput({ DML_TENSOR_DATA_TYPE_FLOAT32, { 1,n} });

	// Constant n [2]
	op1.AddIntermediate(ml.ConstantValueTensor(*op1.GetGraph().get(), (float)n, { 1,1 }));

	// Sx [3]
	op1.AddIntermediate(dml::Slice(dml::CumulativeSummation(op1.Item(0), 1, DML_AXIS_DIRECTION_INCREASING, false), { 0, n - 1 }, { 1, 1 }, { 1, 1 }));

	// Sy [4]
	op1.AddIntermediate(dml::Slice(dml::CumulativeSummation(op1.Item(1), 1, DML_AXIS_DIRECTION_INCREASING, false), { 0, n - 1 }, { 1, 1 }, { 1, 1 }));

	// Sxy [5]
	op1.AddIntermediate(dml::Slice(dml::CumulativeSummation(dml::Multiply(op1.Item(0), op1.Item(1)), 1, DML_AXIS_DIRECTION_INCREASING, false), { 0, n - 1 }, { 1, 1 }, { 1, 1 }));

	// Sx2 [6]
	op1.AddIntermediate(dml::Slice(dml::CumulativeSummation(dml::Multiply(op1.Item(0), op1.Item(0)), 1, DML_AXIS_DIRECTION_INCREASING, false), { 0, n - 1 }, { 1, 1 }, { 1, 1 }));

	// float B = (n * Sxy - Sx * Sy) / ((n * Sx2) - (Sx * Sx));

	// B [7]
	op1.AddOutput((
		dml::Divide(
			dml::Subtract(
				dml::Multiply(
					op1.Item(2),
					op1.Item(5)
				),
				dml::Multiply(
					op1.Item(3),
					op1.Item(4)
				)
			),
			dml::Subtract(
				dml::Multiply(
					op1.Item(2),
					op1.Item(6)
				),
				dml::Multiply(
					op1.Item(3),
					op1.Item(3)
				)
			)
		)
	));

	// float A = (Sy - (B * Sx)) / n; [8]

	op1.AddOutput(dml::Divide(
		dml::Subtract(
			op1.Item(4),
			dml::Multiply(
				op1.Item(7),
				op1.Item(3)
			)
		),
		op1.Item(2)
	));


	ml.ops.push_back(op1.Build());
	ml.Prepare();

	ml.ops[0].Item(0).buffer->Upload(&ml, px, n * sizeof(float));	
	ml.ops[0].Item(1).buffer->Upload(&ml, py, n * sizeof(float));
	
	ml.Run();

	std::vector<char> cdata;
	float A = 0, B = 0;

	ml.ops[0].Item(7).buffer->Download(&ml, (size_t)-1, cdata);
	memcpy(&B, cdata.data(), 4);
	ml.ops[0].Item(8).buffer->Download(&ml, (size_t)-1, cdata);
	memcpy(&A, cdata.data(), 4);

	printf("Linear Regression GPU:\r\nA = %f\r\nB = %f\r\n\r\n", A, B);
}


void LinearRegressionTest()
{
	RandomData();

	// Do it in CPU
	LinearRegressionCPU(xs.data(), ys.data(), N);

	// Do it in DML
	LinearRegressionDML(xs.data(), ys.data(), (unsigned int)N);
}


int main()
{
	CoInitializeEx(NULL, COINIT_MULTITHREADED);
	ML ml(true);
	ml.SetFeatureLevel(DML_FEATURE_LEVEL_6_4);
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

	// Test LR
	LinearRegressionTest();
}

