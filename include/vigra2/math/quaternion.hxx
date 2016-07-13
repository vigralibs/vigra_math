/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2015 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA2 computer vision library.          */
/*    The VIGRA2 Website is                                             */
/*        http://ukoethe.github.io/vigra2                               */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#pragma once

#ifndef VIGRA2_MATH_QUATERNION_HXX
#define VIGRA2_MATH_QUATERNION_HXX

#include <vigra2/numeric_traits.hxx>
#include <vigra2/tinyarray.hxx>
#include <vigra2/mathutil.hxx>
#include <iosfwd>   // ostream


namespace vigra {

/** Quaternion class.

    Quaternions are mainly used as a compact representation for 3D rotations because
    they are much less prone to round-off errors than rotation matrices, especially
    when many rotations are concatenated. In addition, the angle/axis interpretation
    of normalized quaternions is very intuitive. Read the
    <a href="http://en.wikipedia.org/wiki/Quaternion">Wikipedia entry on quaternions</a>
    for more information on the mathematics.

    See also: \ref QuaternionOperations
*/
template<class ValueType>
class Quaternion {
  public:
    typedef TinyArray<ValueType, 3> Vector;

        /** the quaternion's valuetype
        */
    typedef ValueType value_type;

        /** reference (return of operator[]).
        */
    typedef ValueType & reference;

        /** const reference (return of operator[] const).
        */
    typedef ValueType const & const_reference;

        /** Construct a quaternion with explicit values for the real and imaginary parts.
        */
    Quaternion(ValueType w = 0, ValueType x = 0, ValueType y = 0, ValueType z = 0)
    : w_(w), v_(x, y, z)
    {}

        /** Construct a quaternion with real value and imaginary vector.

            Equivalent to <tt>Quaternion(w, v[0], v[1], v[2])</tt>.
        */
    Quaternion(ValueType w, const Vector &v)
    : w_(w), v_(v)
    {}

        /** Copy constructor.
        */
    Quaternion(const Quaternion &q)
    : w_(q.w_), v_(q.v_)
    {}

        /** Copy assignment.
        */
    Quaternion & operator=(Quaternion const & other)
    {
        w_ = other.w_;
        v_ = other.v_;
        return *this;
    }

        /** Assign \a w to the real part and set the imaginary part to zero.
        */
    Quaternion & operator=(ValueType w)
    {
        w_ = w;
        v_.init(0);
        return *this;
    }

        /**
         * Creates a Quaternion which represents the operation of
         * rotating around the given axis by the given angle.
         *
         * The angle should be in the range -pi..3*pi for sensible
         * results.
         */
    static Quaternion
    createRotation(double angle, const Vector &rotationAxis)
    {
        // the natural range would be -pi..pi, but the reflective
        // behavior around pi is too unexpected:
        if(angle > M_PI)
            angle -= 2.0*M_PI;
        double t = sin(angle/2.0);
        return Quaternion(sqrt(1.0-t*t), (t/norm(rotationAxis))*rotationAxis);
    }

        /** Read real part.
        */
    ValueType w() const { return w_; }
        /** Access real part.
        */
    ValueType &w() { return w_; }
        /** Set real part.
        */
    void setW(ValueType w) { w_ = w; }

        /** Read imaginary part.
        */
    const Vector &v() const { return v_; }
        /** Access imaginary part.
        */
    Vector &v() { return v_; }
        /** Set imaginary part.
        */
    void setV(const Vector & v) { v_ = v; }
        /** Set imaginary part.
        */
    void setV(ValueType x, ValueType y, ValueType z)
    {
        v_[0] = x;
        v_[1] = y;
        v_[2] = z;
    }

    ValueType x() const { return v_[0]; }
    ValueType y() const { return v_[1]; }
    ValueType z() const { return v_[2]; }
    ValueType &x() { return v_[0]; }
    ValueType &y() { return v_[1]; }
    ValueType &z() { return v_[2]; }
    void setX(ValueType x) { v_[0] = x; }
    void setY(ValueType y) { v_[1] = y; }
    void setZ(ValueType z) { v_[2] = z; }

        /** Access entry at index (0 <=> w(), 1 <=> v[0] etc.).
        */
    value_type & operator[](int index)
    {
        return (&w_)[index];
    }

        /** Read entry at index (0 <=> w(), 1 <=> v[0] etc.).
        */
    value_type operator[](int index) const
    {
        return (&w_)[index];
    }

        /** Add \a w to the real part.

            If the quaternion represents a rotation, the rotation angle is
            increased by \a w.
        */
    Quaternion &operator+=(value_type const &w)
    {
        w_ += w;
        return *this;
    }

        /** Add assigment.
        */
    Quaternion &operator+=(Quaternion const &other)
    {
        w_ += other.w_;
        v_ += other.v_;
        return *this;
    }

        /** Subtract \a w from the real part.

            If the quaternion represents a rotation, the rotation angle is
            decreased by \a w.
        */
    Quaternion &operator-=(value_type const &w)
    {
        w_ -= w;
        return *this;
    }

        /** Subtract assigment.
        */
    Quaternion &operator-=(Quaternion const &other)
    {
        w_ -= other.w_;
        v_ -= other.v_;
        return *this;
    }

        /** Addition.
        */
    Quaternion operator+() const
    {
        return *this;
    }

        /** Subtraction.
        */
    Quaternion operator-() const
    {
        return Quaternion(-w_, -v_);
    }

        /** Multiply assignment.

            If the quaternions represent rotations, the rotations of <tt>this</tt> and
            \a other are concatenated.
        */
    Quaternion &operator*=(Quaternion const &other)
    {
        value_type newW = w_*other.w_ - dot(v_, other.v_);
        v_              = w_ * other.v_ + other.w_ * v_ + cross(v_, other.v_);
        w_              = newW;
        return *this;
    }

        /** Multiply all entries with the scalar \a scale.
        */
    Quaternion &operator*=(double scale)
    {
        w_ *= scale;
        v_ *= scale;
        return *this;
    }

        /** Divide assignment.
        */
    Quaternion &operator/=(Quaternion const &other)
    {
        (*this) *= conj(other) / squaredNorm(other);
        return *this;
    }

        /** Devide all entries by the scalar \a scale.
        */
    Quaternion &operator/=(double scale)
    {
        w_ /= scale;
        v_ /= scale;
        return *this;
    }

        /** Equal.
        */
    bool operator==(Quaternion const &other) const
    {
      return (w_ == other.w_) && (v_ == other.v_);
    }

        /** Not equal.
        */
    bool operator!=(Quaternion const &other) const
    {
      return (w_ != other.w_) || (v_ != other.v_);
    }

        /**
         * Fill the first 3x3 elements of the given matrix with a
         * rotation matrix performing the same 3D rotation as this
         * quaternion.
         */
    void fillRotationMatrix(TinyArray<value_type, 3, 3> & matrix) const
    {
        Vector
            vs = v_ * (2.0 / squaredNorm(*this)),
            wv = w_ * vs,
            vv = vs * v_;
        value_type
            xy = vs[0] * v_[1],
            xz = vs[0] * v_[2],
            yz = vs[1] * v_[2];

        matrix(0, 0) = 1 - (vv[1] + vv[2]);
        matrix(0, 1) =     ( xy   - wv[2]);
        matrix(0, 2) =     ( xz   + wv[1]);

        matrix(1, 0) =     ( xy   + wv[2]);
        matrix(1, 1) = 1 - (vv[0] + vv[2]);
        matrix(1, 2) =     ( yz   - wv[0]);

        matrix(2, 0) =     ( xz   - wv[1]);
        matrix(2, 1) =     ( yz   + wv[0]);
        matrix(2, 2) = 1 - (vv[0] + vv[1]);
    }

  protected:
    ValueType w_;
    Vector v_;
};

/** \defgroup QuaternionOperations Quaternion Operations
*/
//@{

    /// Create conjugate quaternion.
template<class ValueType>
Quaternion<ValueType> conj(Quaternion<ValueType> const & q)
{
    return Quaternion<ValueType>(q.w(), -q.v());
}

    /// Addition.
template<typename Type>
inline Quaternion<Type>
operator+(const Quaternion<Type>& t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) += t2;
}

    /// Addition of a scalar on the right.
template<typename Type>
inline Quaternion<Type>
operator+(const Quaternion<Type>& t1,
          const Type& t2)
{
  return Quaternion<Type>(t1) += t2;
}

    /// Addition of a scalar on the left.
template<typename Type>
inline Quaternion<Type>
operator+(const Type& t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) += t2;
}

    /// Subtraction.
template<typename Type>
inline Quaternion<Type>
operator-(const Quaternion<Type>& t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) -= t2;
}

    /// Subtraction of a scalar on the right.
template<typename Type>
inline Quaternion<Type>
operator-(const Quaternion<Type>& t1,
          const Type& t2)
{
  return Quaternion<Type>(t1) -= t2;
}

    /// Subtraction of a scalar on the left.
template<typename Type>
inline Quaternion<Type>
operator-(const Type& t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) -= t2;
}

    /// Multiplication.
template<typename Type>
inline Quaternion<Type>
operator*(const Quaternion<Type>& t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) *= t2;
}

    /// Multiplication with a scalar on the right.
template<typename Type>
inline Quaternion<Type>
operator*(const Quaternion<Type>& t1,
          double t2)
{
  return Quaternion<Type>(t1) *= t2;
}

    /// Multiplication with a scalar on the left.
template<typename Type>
inline Quaternion<Type>
operator*(double t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) *= t2;
}

    /// Division
template<typename Type>
inline Quaternion<Type>
operator/(const Quaternion<Type>& t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) /= t2;
}

    /// Division by a scalar.
template<typename Type>
inline Quaternion<Type>
operator/(const Quaternion<Type>& t1,
          double t2)
{
  return Quaternion<Type>(t1) /= t2;
}

    /// Division of a scalar by a Quaternion.
template<typename Type>
inline Quaternion<Type>
operator/(double t1,
          const Quaternion<Type>& t2)
{
  return Quaternion<Type>(t1) /= t2;
}

    /// squared norm
template <class Type>
inline
SquaredNormType<Quaternion<Type> >
squaredNorm(Quaternion<Type> const & q)
{
    return sq(q.w()) + squaredNorm(q.v());
}

    /// norm
template<typename Type>
inline
NormType<Quaternion<Type> >
abs(Quaternion<Type> const & q)
{
    return norm(q);
}

//@}

} // namespace vigra

namespace std {

template<class ValueType>
inline
ostream & operator<<(ostream & os, vigra::Quaternion<ValueType> const & q)
{
    os << q.w() << " " << q.x() << " " << q.y() << " " << q.z();
    return os;
}

template<class ValueType>
inline
istream & operator>>(istream & is, vigra::Quaternion<ValueType> & q)
{
    ValueType w, x, y, z;
    is >> w >> x >> y >> z;
    q.setW(w);
    q.setX(x);
    q.setY(y);
    q.setZ(z);
    return is;
}

} // namespace std

#endif // VIGRA2_MATH_QUATERNION_HXX
