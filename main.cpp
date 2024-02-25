#include<mpi.h>
#include<vector>
#include<iostream>
#include<chrono>
#include <math.h>


const double t_plus = 0.0001;
const double t_minus = -0.0001;
const double epsilon = 0.001;

class Matrix
{
private:
	std::vector< std::vector<double> > matrix;
	int N;

public:
	Matrix(int N) 
	{

		N = N;
		matrix.resize(N, std::vector<double>(N, 1.0));
		for (int i = 0; i < matrix.size(); i++)
		{
			for (int j = 0; j < matrix.size(); j++)
			{
				if (i == j)
				{
					matrix[i][j] = 2.0;
				}
			}
		}
	}
	int size() {
		return N;
	}
	const std::vector<double>& operator[](int index) const;
	std::vector<double>& operator[](int index);
};

std::vector<double>& Matrix::operator[](int index) {
	return matrix[index];
}

const std::vector<double>& Matrix::operator[](int index) const {
	return matrix[index];
}


double find_norm(const std::vector<double>& row)
{

	int size, rank;
	double norm = 0;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	int count_for_process = ceil((double)row.size() / size);
	double result = 0.0;
	for (int i = 0; i < count_for_process; i++)
	{
		result += row[i] * row[i];
	}

	MPI_Allreduce(&result, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	norm = pow(norm, 0.5);
	return norm;
}

double multiply_row_by_column(const std::vector<double>& row, const std::vector<double>& column) 
{
	double result = 0.0;
	for (int i = 0; i < row.size(); ++i) {
		result += row[i] * column[i];
	}
	return result;
}

bool check(Matrix A, std::vector<double>& x, std::vector<double>& b, std::vector<double>& result, int N, double b_norm)
{
	//const double ep = epsilon * epsilon;

	int size, rank;


	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	int count_for_process = ceil((double)N / size);
	int ibeg = count_for_process * rank;
	int iend = count_for_process * (rank + 1) > N ? N : count_for_process * (rank + 1);

	for (int i = ibeg; i < iend; i++)
	{
		result[i] = multiply_row_by_column(A[i], x) - b[i];
	}



	double norm_numerator = find_norm(result);
	return norm_numerator / b_norm < epsilon;
}


void  proximity(Matrix A, std::vector<double> &x, std::vector<double> &b, std::vector<double>& x_process, int N)
{
	int size, rank;


	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	int count_for_process = ceil((double)N / size);
	int ibeg = count_for_process * rank;
	int iend = count_for_process * (rank + 1) > N ? N : count_for_process * (rank + 1);

	for (int i = ibeg, j = 0; i < iend; i++, j++) 
	{
		x_process[j] = multiply_row_by_column(A[i], x);
		x_process[j] = x[i] - t_plus * (x_process[j] - b[i]);
	}

	
	//MPI_Gather(&x_process[0], count_for_process, MPI_DOUBLE, &x[0], count_for_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if (rank == 0)
	{
		for (int i = 0; i < count_for_process; i++)
		{
			x[i] = x_process[i];
		}
		for (int i = 1; i < size; i++)
		{
			MPI_Recv(&x[i * count_for_process], count_for_process, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		for (int i = 1; i < size; i++)
		{
			MPI_Ssend(&x[0], x.size(), MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}


	}
	else
	{
		MPI_Ssend(&x_process[0], iend - ibeg, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(&x[0], x.size(), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	/*if (rank == 0) {

		

		/*for (int i = 0; i < count_for_process; i++)
		{
			x[i] = x_process[i];
		}
		MPI_Gather(x_process.data(), count_for_process, MPI_DOUBLE, x.data(), count_for_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	else {
		MPI_Gather(x_process.data(), count_for_process, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}*/

}


int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	int N = 1000;
	Matrix A(N);
	std::vector<double> b(N, N + 1);
	double b_norm = find_norm(b);

	std::vector<double> result(N);

	int size, rank;

	std::vector<double> x_process(N);

	std::vector<double> x(N, 2.0);

	auto start_time2 = MPI_Wtime();


	while (!check(A, x, b, result, N, b_norm)) {

		proximity(A, x, b, x_process, N);
		//std::cout << x[0] << std::endl;
	}

	auto end_time2 = MPI_Wtime();
	auto duration2 = end_time2 - start_time2;

	std::cout << "Time passed: " << duration2 << std::endl;
	
	

	for (int i = 0; i < x.size(); i++) {

		std::cout << x[i] << " ";
	}
	std::cout << std::endl;


	MPI_Finalize();


	return 0;
}
