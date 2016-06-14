////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// @file    include/nbfmm/utility/operator.hpp
/// @brief   The operators for CUDA types
///
/// @author  Mu Yang <emfomy@gmail.com>
///

#ifndef NBFMM_UTILTITY_OPERATOR_HPP_
#define NBFMM_UTILTITY_OPERATOR_HPP_

#include <nbfmm/config.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  int2 functions
//

/// Addition operator
__host__ __device__  inline int2 operator+( const int2 a, const int2 b ) {
  return make_int2( a.x + b.x, a.y + b.y );
}

/// Addition assignment operator
__host__ __device__  inline void operator+=( int2& a, const int2 b ) {
  a.x += b.x; a.y += b.y;
}

/// Subtract operator
__host__ __device__  inline int2 operator-( const int2 a, const int2 b ) {
  return make_int2( a.x - b.x, a.y - b.y );
}

/// Subtract assignment operator
__host__ __device__  inline void operator-=( int2& a, const int2 b ) {
  a.x -= b.x; a.y -= b.y;
}

/// Multiplication operator
//@{
__host__ __device__  inline int2 operator*( const int2 a, const int2 b ) {
  return make_int2( a.x * b.x, a.y * b.y );
}
__host__ __device__  inline int2 operator*( const int2 a, const int b ) {
  return make_int2( a.x * b, a.y * b );
}
__host__ __device__  inline int2 operator*( const int a, const int2 b ) {
  return make_int2( a * b.x, a * b.y );
}
//@}

/// Multiplication assignment operator
//@{
__host__ __device__  inline void operator*=( int2& a, const int2 b ) {
  a.x *= b.x; a.y *= b.y;
}
__host__ __device__  inline void operator*=( int2& a, const int b ) {
  a.x *= b; a.y *= b;
}
//@}

/// Division operator
//@{
__host__ __device__  inline int2 operator/( const int2 a, const int2 b ) {
  return make_int2( a.x / b.x, a.y / b.y );
}
__host__ __device__  inline int2 operator/( const int2 a, const int b ) {
  return make_int2( a.x / b, a.y / b );
}
__host__ __device__  inline int2 operator/( const int a, const int2 b ) {
  return make_int2( a / b.x, a / b.y );
}
//@}

/// Division assignment operator
//@{
__host__ __device__  inline void operator/=( int2& a, const int2 b ) {
  a.x /= b.x; a.y /= b.y;
}
__host__ __device__  inline void operator/=( int2& a, const int b ) {
  a.x /= b; a.y /= b;
}
//@}

/// Equal to operator
__host__ __device__  inline bool operator==( const int2 a, const int2 b ) {
  return (a.x == b.x) && (a.y == b.y);
}

/// Not equal to operator
__host__ __device__  inline bool operator!=( const int2 a, const int2 b ) {
  return (a.x != b.x) || (a.y != b.y);
}

/// Greater than operator (row major)
__host__ __device__  inline bool operator>( const int2 a, const int2 b ) {
  return (a.y != b.y) ? (a.y > b.y) : (a.x > b.x);
}

/// Greater than or equal to operator (row major)
__host__ __device__  inline bool operator>=( const int2 a, const int2 b ) {
  return (a.y != b.y) ? (a.y >= b.y) : (a.x >= b.x);
}

/// Less than operator (row major)
__host__ __device__  inline bool operator<( const int2 a, const int2 b ) {
  return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x);
}

/// Less than or equal to operator (row major)
__host__ __device__  inline bool operator<=( const int2 a, const int2 b ) {
  return (a.y != b.y) ? (a.y <= b.y) : (a.x <= b.x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  float2 functions
//

/// Addition operator
__host__ __device__  inline float2 operator+( const float2 a, const float2 b ) {
  return make_float2( a.x + b.x, a.y + b.y );
}

/// Addition assignment operator
__host__ __device__  inline void operator+=( float2& a, const float2 b ) {
  a.x += b.x; a.y += b.y;
}

/// Subtract operator
__host__ __device__  inline float2 operator-( const float2 a, const float2 b ) {
  return make_float2( a.x - b.x, a.y - b.y );
}

/// Subtract assignment operator
__host__ __device__  inline void operator-=( float2& a, const float2 b ) {
  a.x -= b.x; a.y -= b.y;
}

/// Multiplication operator
//@{
__host__ __device__  inline float2 operator*( const float2 a, const float2 b ) {
  return make_float2( a.x * b.x, a.y * b.y );
}
__host__ __device__  inline float2 operator*( const float2 a, const float b ) {
  return make_float2( a.x * b, a.y * b );
}
__host__ __device__  inline float2 operator*( const float a, const float2 b ) {
  return make_float2( a * b.x, a * b.y );
}
//@}

/// Multiplication assignment operator
//@{
__host__ __device__  inline void operator*=( float2& a, const float2 b ) {
  a.x *= b.x; a.y *= b.y;
}
__host__ __device__  inline void operator*=( float2& a, const float b ) {
  a.x *= b; a.y *= b;
}
//@}

/// Division operator
//@{
__host__ __device__  inline float2 operator/( const float2 a, const float2 b ) {
  return make_float2( a.x / b.x, a.y / b.y );
}
__host__ __device__  inline float2 operator/( const float2 a, const float b ) {
  return make_float2( a.x / b, a.y / b );
}
__host__ __device__  inline float2 operator/( const float a, const float2 b ) {
  return make_float2( a / b.x, a / b.y );
}
//@}

/// Division assignment operator
//@{
__host__ __device__  inline void operator/=( float2& a, const float2 b ) {
  a.x /= b.x; a.y /= b.y;
}
__host__ __device__  inline void operator/=( float2& a, const float b ) {
  a.x /= b; a.y /= b;
}
//@}

/// Equal to operator
__host__ __device__  inline bool operator==( const float2 a, const float2 b ) {
  return (a.x == b.x) && (a.y == b.y);
}

/// Not equal to operator
__host__ __device__  inline bool operator!=( const float2 a, const float2 b ) {
  return (a.x != b.x) || (a.y != b.y);
}

/// Greater than operator (row major)
__host__ __device__  inline bool operator>( const float2 a, const float2 b ) {
  return (a.y != b.y) ? (a.y > b.y) : (a.x > b.x);
}

/// Greater than or equal to operator (row major)
__host__ __device__  inline bool operator>=( const float2 a, const float2 b ) {
  return (a.y != b.y) ? (a.y >= b.y) : (a.x >= b.x);
}

/// Less than operator (row major)
__host__ __device__  inline bool operator<( const float2 a, const float2 b ) {
  return (a.y != b.y) ? (a.y < b.y) : (a.x < b.x);
}

/// Less than or equal to operator (row major)
__host__ __device__  inline bool operator<=( const float2 a, const float2 b ) {
  return (a.y != b.y) ? (a.y <= b.y) : (a.x <= b.x);
}

#endif  // NBFMM_UTILTITY_OPERATOR_HPP_
