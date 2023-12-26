#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

#include <iostream>
#include <random>

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

std::pair<SpMat, Eigen::VectorXd> generateSparseMatrixAndRightSide(const int n) {
  Eigen::VectorXd b(n);
  Eigen::SparseMatrix<double> A(n, n);

  constexpr int numberNonZeroCoeffs = 1000;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, n - 1);
  for (int k = 0; k < numberNonZeroCoeffs; ++k) {
    const int i = dis(gen);
    const int j = dis(gen);
    A.coeffRef(i, j) = dis(gen);
  }

  for (int k = 0; k < n; ++k) {
    b[k] = dis(gen);
  }

  return std::make_pair(A, b);
}

void runSparseMatricesSolvers()
{
  constexpr int n = 10000;
  std::pair<SpMat, Eigen::VectorXd> sle = generateSparseMatrixAndRightSide(n);
  auto A = sle.first;
  auto b = sle.second;
  Eigen::VectorXd x(n);

  // BiCGSTAB
  int t = omp_get_max_threads();
  omp_set_num_threads(t);
  Eigen::setNbThreads(t);
  Eigen::BiCGSTAB<SpMat> solver;

  double start = omp_get_wtime();
  solver.compute(A);
  x = solver.solve(b);
  double end = omp_get_wtime();

  std::cout << "BiCGSTAB:" << std::endl;
  std::cout << "elapsed time: " << end - start << std::endl;
  std::cout << "#iterations:     " << solver.iterations() << std::endl;
  std::cout << "estimated error: " << solver.error() << std::endl;

  // GMRES
  omp_set_num_threads(t);
  Eigen::setNbThreads(t);
  Eigen::GMRES<SpMat> gmres;
  start = omp_get_wtime();
  gmres.compute(A);
  x = gmres.solve(b);
  end = omp_get_wtime();

  std::cout << "GMRES:" << std::endl;
  std::cout << "elapsed time: " << end - start << std::endl;
  std::cout << "#iterations:     " << gmres.iterations() << std::endl;
  std::cout << "estimated error: " << gmres.error() << std::endl;
}

void runDenseMatricesSolvers()
{
  constexpr int n = 2000;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::VectorXd b = Eigen::VectorXd::Random(n);
  Eigen::VectorXd x(n);

  // LU Decomposition
  int t = omp_get_max_threads();
  omp_set_num_threads(t);
  Eigen::setNbThreads(t);
  Eigen::FullPivLU<Eigen::MatrixXd> luSolver(A);

  double start = omp_get_wtime();
  x = luSolver.solve(b);
  double end = omp_get_wtime();

  std::cout << "LU Decomposition:" << std::endl;
  std::cout << "elapsed time: " << end - start << std::endl;
  std::cout << "Rank: " << luSolver.rank() << std::endl;

  // QR Decomposition
  omp_set_num_threads(t);
  Eigen::setNbThreads(t);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrSolver(A);

  start = omp_get_wtime();
  x = qrSolver.solve(b);
  end = omp_get_wtime();

  std::cout << "QR Decomposition:" << std::endl;
  std::cout << "elapsed time: " << end - start << std::endl;
  std::cout << "Rank: " << qrSolver.rank() << std::endl;

}

void runSymmetricMatricesSolvers()
{
  constexpr int n = 5000;

  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd symmetricA = A.transpose() * A;

  Eigen::VectorXd b = Eigen::VectorXd::Random(n);
  Eigen::VectorXd x;

  // Cholesky Decomposition
  Eigen::LLT<Eigen::MatrixXd> llt;
  int t = omp_get_max_threads();
  omp_set_num_threads(t);
  Eigen::setNbThreads(t);

  double start = omp_get_wtime();
  llt.compute(symmetricA);
  x = llt.solve(b);
  double end = omp_get_wtime();

	std::cout << "Cholesky decomposition:" << std::endl;
  std::cout << "elapsed time: " << end - start << std::endl;
}

int main()
{
  runSparseMatricesSolvers();
  runDenseMatricesSolvers();
  runSymmetricMatricesSolvers();

	return 0;
}