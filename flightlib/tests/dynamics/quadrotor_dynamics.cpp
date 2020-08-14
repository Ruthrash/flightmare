#include "flightlib/dynamics/quadrotor_dynamics.hpp"

#include <gtest/gtest.h>

#include "flightlib/common/math.hpp"
#include "flightlib/common/quad_state.hpp"

using namespace flightlib;

static constexpr Scalar MASS = 1.0;
static constexpr Scalar ARM_LENGTH = 0.25;

TEST(QuadrotorDynamics, Constructor) {
  QuadrotorDynamics quad(MASS, ARM_LENGTH);

  QuadrotorDynamics quad_copy(quad);

  Scalar mass = quad_copy.mass();
  Matrix<3, 3> J = quad_copy.J();
  Matrix<3, 3> J_inv = quad_copy.J_inv();

  std::cout << J << std::endl;

  Matrix<3, 3> expected_J = (mass / 12.0 * ARM_LENGTH * ARM_LENGTH *
                             Vector<3>(2.25, 2.25, 4).asDiagonal());
  Matrix<3, 3> expected_J_inv = expected_J.inverse();

  EXPECT_EQ(mass, MASS);
  EXPECT_TRUE(J.isApprox(expected_J));
  EXPECT_TRUE(J_inv.isApprox(expected_J_inv));
}

TEST(QuadrotorDynamics, Dynamics) {
  QuadrotorDynamics quad(MASS, ARM_LENGTH);

  QuadState hover;
  hover.setZero();

  Vector<> derivative(hover.x);
  QuadState derivative_state;
  derivative_state.setZero();

  EXPECT_TRUE(quad.dState(hover.x, derivative));
  EXPECT_TRUE(quad.dState(hover, &derivative_state));

  EXPECT_TRUE(derivative.isZero());
  EXPECT_TRUE(derivative_state.x.isZero());

  static constexpr int N = 128;

  for (int trail = 0; trail < N; ++trail) {
    // Generate a random state.
    QuadState random_state;
    random_state.x = Vector<QuadState::SIZE>::Random();
    random_state.qx.normalize();

    // Get the differential.
    EXPECT_TRUE(quad.dState(random_state, &derivative_state));

    // Compute it manually.
    QuadState derivative_manual;
    derivative_manual.setZero();

    const Quaternion q_omega(0, random_state.w.x(), random_state.w.y(),
                             random_state.w.z());
    derivative_manual.p = random_state.v;
    derivative_manual.qx = 0.5 * Q_right(q_omega) * random_state.qx;
    derivative_manual.v = random_state.a;
    derivative_manual.w =
      quad.J_inv() *
      (random_state.tau - random_state.w.cross(quad.J() * random_state.w));

    // Compare the derivatives.
    EXPECT_TRUE(derivative_state.x.isApprox(derivative_manual.x));
  }
}

TEST(QuadrotorDynamics, VectorReference) {
  const QuadrotorDynamics quad(MASS, ARM_LENGTH);

  static constexpr int N = 128;
  Matrix<QuadState::SIZE, N> states = Matrix<QuadState::SIZE, N>::Random();
  Matrix<QuadState::SIZE, N> states_const =
    Matrix<QuadState::SIZE, N>::Random();
  Matrix<QuadState::SIZE, N> derivates;

  for (int i = 0; i < N; ++i) {
    QuadState derivate;
    EXPECT_TRUE(quad.dState(states.col(i), derivates.col(i)));
    EXPECT_TRUE(quad.dState(QuadState(states.col(i)), &derivate));
    EXPECT_TRUE(quad.dState(states_const.col(i), derivates.col(i)));
  }
}