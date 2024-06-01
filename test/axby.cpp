#include <glog/logging.h>
#include <gtest/gtest.h>

#include <armadillo>
#include <iostream>

void Axby(const arma::fmat &x, const arma::fmat &w, const arma::fmat &b,
          arma::fmat &y) {
  // y = w * x + b
  y = w * x + b;

  std::cout << y << std::endl;
}

TEST(test_arma, Axby) {
  using namespace arma;
  fmat w =
      "1,2,3;"
      "4,5,6;"
      "7,8,9;";

  fmat x =
      "1,2,3;"
      "4,5,6;"
      "7,8,9;";

  fmat b =
      "1,1,1;"
      "2,2,2;"
      "3,3,3;";

  fmat answer =
      "31,37,43;"
      "68,83,98;"
      "105,129,153";

  fmat y;
  Axby(x, w, b, y);
  ASSERT_EQ(approx_equal(y, answer, "absdiff", 1e-5f), true);
}

void EPowerMinus(const arma::fmat &x, arma::fmat &y) {
  // y = e^{-x}
  y = arma::exp(-x);
}

TEST(test_arma, e_power_minus) {
  using namespace arma;

  fmat x(224, 224);
  x.randu();

  fmat y;
  EPowerMinus(x, y);
  std::vector<float> x1(x.mem, x.mem + 224 * 224);
  ASSERT_EQ(y.empty(), false);
  for (int i = 0; i < 224 * 224; ++i) {
    ASSERT_LE(std::abs(std::exp(-x1.at(i)) - y.at(i)), 1e-5f);
  }
}

void Axpy(const arma::fmat &x, arma::fmat &Y, float a, float y) {
  // Y = a * x + y
  Y = a * x + y;
}

TEST(test_arma, axpy) {
  using namespace arma;
  fmat x(224, 224);
  x.randu();

  fmat Y;
  float a = 3.f;
  float y = 4.f;
  Axpy(x, Y, a, y);

  ASSERT_EQ(Y.empty(), false);
  std::vector<float> x1(x.mem, x.mem + 224 * 224);
  for (int i = 0; i < 224 * 224; ++i) {
    ASSERT_LE(std::abs(x.at(i) * a + y - Y.at(i)), 1e-5f);
  }
}