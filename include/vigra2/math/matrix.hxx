/************************************************************************/
/*                                                                      */
/*               Copyright 2014-2016 by Ullrich Koethe                  */
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

#ifndef VIGRA2_MATH_MATRIX_HXX
#define VIGRA2_MATH_MATRIX_HXX

#include <cmath>
#include <iosfwd>
#include <iomanip>
#include <vigra2/array_nd.hxx>
#include <vigra2/mathutil.hxx>
#include <vigra2/numeric_traits.hxx>
#include <vigra2/algorithm.hxx>

namespace vigra
{

/** \defgroup LinearAlgebraModule Linear Algebra

    \brief Classes and functions for matrix algebra, linear equations systems, eigen systems, least squares etc.
*/

/** \ingroup LinearAlgebraModule

    Namespace <tt>vigra2/linalg</tt> hold VIGRA's linear algebra functionality. But most of its contents
    is exported into namespace <tt>vigra</tt> via <tt>using</tt> directives.
*/
namespace linalg
{

template <class T>
inline ArrayIndex
rowCount(const ArrayViewND<2, T> &x);

template <class T>
inline ArrayIndex
columnCount(const ArrayViewND<2, T> &x);

template <class T>
inline ArrayViewND <2, T>
rowVector(ArrayViewND <2, T> const & m, ArrayIndex d);

template <class T>
inline ArrayViewND <2, T>
columnVector(ArrayViewND<2, T> const & m, ArrayIndex d);

// template <class T, class ALLOC = std::allocator<T> >
// class TemporaryMatrix;

template <class T>
void transpose(const ArrayViewND<2, T> &v, ArrayViewND<2, T> &r);

template <class T>
bool isSymmetric(const ArrayViewND<2, T> &v);

template <class T>
using Matrix = ArrayND<2, T>;

using vigra::RowMajor;
using vigra::ColumnMajor;

/********************************************************/
/*                                                      */
/*                        Matrix                        */
/*                                                      */
/********************************************************/

/** Matrix class.

    \ingroup LinearAlgebraModule

    This is the basic class for all linear algebra computations. Matrices are
    stored in a <i>column-major</i> format, i.e. the row index is varying fastest.
    This is the same format as in the lapack and gmm++ libraries, so it will
    be easy to interface these libraries. In fact, if you need optimized
    high performance code, you should use them. The VIGRA linear algebra
    functionality is provided for smaller problems and rapid prototyping
    (no one wants to spend half the day installing a new library just to
    discover that the new algorithm idea didn't work anyway).

    <b>See also:</b>
    <ul>
    <li> \ref LinearAlgebraFunctions
    </ul>

    <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
    <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
    Namespaces: vigra and vigra::linalg
*/
#if 0
template <class T, class ALLOC = std::allocator<T> >
class Matrix
: public MultiArray<2, T, ALLOC>
{
    typedef MultiArray<2, T, ALLOC> BaseType;

  public:
    typedef Matrix<T, ALLOC>                        matrix_type;
    typedef TemporaryMatrix<T, ALLOC>               temp_type;
    typedef ArrayViewND<2, T>                    view_type;
    typedef typename BaseType::value_type           value_type;
    typedef typename BaseType::pointer              pointer;
    typedef typename BaseType::const_pointer        const_pointer;
    typedef typename BaseType::reference            reference;
    typedef typename BaseType::const_reference      const_reference;
    typedef typename BaseType::difference_type      difference_type;
    typedef typename BaseType::difference_type_1    difference_type_1;
    typedef ALLOC                                   allocator_type;

        /** default constructor
         */
    Matrix()
    {}

        /** construct with given allocator
         */
    explicit Matrix(ALLOC const & alloc)
    : BaseType(alloc)
    {}

        /** construct with given shape and init all
            elements with zero. Note that the order of the axes is
            <tt>difference_type(rows, columns)</tt> which
            is the opposite of the usual VIGRA convention.
         */
    explicit Matrix(const difference_type &aShape,
                    ALLOC const & alloc = allocator_type())
    : BaseType(aShape, alloc)
    {}

        /** construct with given shape and init all
            elements with zero. Note that the order of the axes is
            <tt>(rows, columns)</tt> which
            is the opposite of the usual VIGRA convention.
         */
    Matrix(difference_type_1 rows, difference_type_1 columns,
                    ALLOC const & alloc = allocator_type())
    : BaseType(difference_type(rows, columns), alloc)
    {}

        /** construct with given shape and init all
            elements with the constant \a init. Note that the order of the axes is
            <tt>difference_type(rows, columns)</tt> which
            is the opposite of the usual VIGRA convention.
         */
    Matrix(const difference_type &aShape, const_reference init,
           allocator_type const & alloc = allocator_type())
    : BaseType(aShape, init, alloc)
    {}

        /** construct with given shape and init all
            elements with the constant \a init. Note that the order of the axes is
            <tt>(rows, columns)</tt> which
            is the opposite of the usual VIGRA convention.
         */
    Matrix(difference_type_1 rows, difference_type_1 columns, const_reference init,
           allocator_type const & alloc = allocator_type())
    : BaseType(difference_type(rows, columns), init, alloc)
    {}

        /** construct with given shape and copy data from C-style array \a init.
            Unless \a layout is <tt>ColumnMajor</tt>, the elements in this array
            are assumed to be given in row-major order (the C standard order) and
            will automatically be converted to the required column-major format.
            Note that the order of the axes is <tt>difference_type(rows, columns)</tt> which
            is the opposite of the usual VIGRA convention.
         */
    Matrix(const difference_type &shape, const_pointer init, RawArrayMemoryLayout layout = RowMajor,
           allocator_type const & alloc = allocator_type())
    : BaseType(shape, alloc) // FIXME: this function initializes the memory twice
    {
        if(layout == RowMajor)
        {
            difference_type trans(shape[1], shape[0]);
            linalg::transpose(ArrayViewND<2, T>(trans, const_cast<pointer>(init)), *this);
        }
        else
        {
            std::copy(init, init + elementCount(), this->data());
        }
    }

        /** construct with given shape and copy data from C-style array \a init.
            Unless \a layout is <tt>ColumnMajor</tt>, the elements in this array
            are assumed to be given in row-major order (the C standard order) and
            will automatically be converted to the required column-major format.
            Note that the order of the axes is <tt>(rows, columns)</tt> which
            is the opposite of the usual VIGRA convention.
         */
    Matrix(difference_type_1 rows, difference_type_1 columns, const_pointer init, RawArrayMemoryLayout layout = RowMajor,
           allocator_type const & alloc = allocator_type())
    : BaseType(difference_type(rows, columns), alloc) // FIXME: this function initializes the memory twice
    {
        if(layout == RowMajor)
        {
            difference_type trans(columns, rows);
            linalg::transpose(ArrayViewND<2, T>(trans, const_cast<pointer>(init)), *this);
        }
        else
        {
            std::copy(init, init + elementCount(), this->data());
        }
    }

        /** copy constructor. Allocates new memory and
            copies tha data.
         */
    Matrix(const Matrix &rhs)
    : BaseType(rhs)
    {}

        /** construct from temporary matrix, which looses its data.

            This operation is equivalent to
            \code
                TemporaryMatrix<T> temp = ...;

                Matrix<T> m;
                m.swap(temp);
            \endcode
         */
    Matrix(const TemporaryMatrix<T, ALLOC> &rhs)
    : BaseType(rhs.allocator())
    {
        this->swap(const_cast<TemporaryMatrix<T, ALLOC> &>(rhs));
    }

        /** construct from a ArrayViewND. Allocates new memory and
            copies tha data. \a rhs is assumed to be in column-major order already.
         */
    template<class U>
    Matrix(const ArrayViewND<2, U> &rhs)
    : BaseType(rhs)
    {}

        /** assignment.
            If the size of \a rhs is the same as the matrix's old size, only the data
            are copied. Otherwise, new storage is allocated, which invalidates
            all objects (array views, iterators) depending on the matrix.
         */
    Matrix & operator=(const Matrix &rhs)
    {
        BaseType::operator=(rhs); // has the correct semantics already
        return *this;
    }

        /** assign a temporary matrix. If the shapes of the two matrices match,
            only the data are copied (in order to not invalidate views and iterators
            depending on this matrix). Otherwise, the memory is swapped
            between the two matrices, so that all depending objects
            (array views, iterators) ar invalidated.
         */
    Matrix & operator=(const TemporaryMatrix<T, ALLOC> &rhs)
    {
        if(this->shape() == rhs.shape())
            this->copy(rhs);
        else
            this->swap(const_cast<TemporaryMatrix<T, ALLOC> &>(rhs));
        return *this;
    }

        /** assignment from arbitrary 2-dimensional ArrayViewND.<br>
            If the size of \a rhs is the same as the matrix's old size, only the data
            are copied. Otherwise, new storage is allocated, which invalidates
            all objects (array views, iterators) depending on the matrix.
            \a rhs is assumed to be in column-major order already.
         */
    template <class U>
    Matrix & operator=(const ArrayViewND<2, U> &rhs)
    {
        BaseType::operator=(rhs); // has the correct semantics already
        return *this;
    }

        /** assignment from scalar.<br>
            Equivalent to Matrix::init(v).
         */
    Matrix & operator=(value_type const & v)
    {
        return init(v);
    }

         /** init elements with a constant
         */
    template <class U>
    Matrix & init(const U & init)
    {
        BaseType::init(init);
        return *this;
    }

       /** reshape to the given shape and initialize with zero.
         */
    void reshape(difference_type_1 rows, difference_type_1 columns)
    {
        BaseType::reshape(difference_type(rows, columns));
    }

        /** reshape to the given shape and initialize with \a init.
         */
    void reshape(difference_type_1 rows, difference_type_1 columns, const_reference init)
    {
        BaseType::reshape(difference_type(rows, columns), init);
    }

        /** reshape to the given shape and initialize with zero.
         */
    void reshape(difference_type const & newShape)
    {
        BaseType::reshape(newShape);
    }

        /** reshape to the given shape and initialize with \a init.
         */
    void reshape(difference_type const & newShape, const_reference init)
    {
        BaseType::reshape(newShape, init);
    }

        /** Create a matrix view that represents the row vector of row \a d.
         */
    view_type rowVector(difference_type_1 d) const
    {
        return vigra::linalg::rowVector(*this, d);
    }

        /** Create a matrix view that represents the column vector of column \a d.
         */
    view_type columnVector(difference_type_1 d) const
    {
        return vigra::linalg::columnVector(*this, d);
    }

        /** number of rows (height) of the matrix.
        */
    difference_type_1 rowCount() const
    {
        return this->m_shape[0];
    }

        /** number of columns (width) of the matrix.
        */
    difference_type_1 columnCount() const
    {
        return this->m_shape[1];
    }

        /** number of elements (width*height) of the matrix.
        */
    difference_type_1 elementCount() const
    {
        return rowCount()*columnCount();
    }

        /** check whether the matrix is symmetric.
        */
    bool isSymmetric() const
    {
        return vigra::linalg::isSymmetric(*this);
    }

        /** sums over the matrix.
        */
    TemporaryMatrix<T> sum() const
    {
        TemporaryMatrix<T> result(1, 1);
        vigra::transformMultiArray(srcMultiArrayRange(*this),
                               destMultiArrayRange(result),
                               vigra::FindSum<T>() );
        return result;
    }

        /** sums over dimension \a d of the matrix.
        */
    TemporaryMatrix<T> sum(difference_type_1 d) const
    {
        difference_type shape(d==0 ? 1 : this->m_shape[0], d==0 ? this->m_shape[1] : 1);
        TemporaryMatrix<T> result(shape);
        vigra::transformMultiArray(srcMultiArrayRange(*this),
                               destMultiArrayRange(result),
                               vigra::FindSum<T>() );
        return result;
    }

        /** sums over the matrix.
        */
    TemporaryMatrix<T> mean() const
    {
        TemporaryMatrix<T> result(1, 1);
        vigra::transformMultiArray(srcMultiArrayRange(*this),
                               destMultiArrayRange(result),
                               vigra::FindAverage<T>() );
        return result;
    }

        /** calculates mean over dimension \a d of the matrix.
        */
    TemporaryMatrix<T> mean(difference_type_1 d) const
    {
        difference_type shape(d==0 ? 1 : this->m_shape[0], d==0 ? this->m_shape[1] : 1);
        TemporaryMatrix<T> result(shape);
        vigra::transformMultiArray(srcMultiArrayRange(*this),
                               destMultiArrayRange(result),
                               vigra::FindAverage<T>() );
        return result;
    }


#ifdef DOXYGEN
// repeat the following functions for documentation. In real code, they are inherited.

        /** read/write access to matrix element <tt>(row, column)</tt>.
            Note that the order of the argument is the opposite of the usual
            VIGRA convention due to column-major matrix order.
        */
    value_type & operator()(difference_type_1 row, difference_type_1 column);

        /** read access to matrix element <tt>(row, column)</tt>.
            Note that the order of the argument is the opposite of the usual
            VIGRA convention due to column-major matrix order.
        */
    value_type operator()(difference_type_1 row, difference_type_1 column) const;

        /** squared Frobenius norm. Sum of squares of the matrix elements.
        */
    typename NormTraits<Matrix>::SquaredNormType squaredNorm() const;

        /** Frobenius norm. Root of sum of squares of the matrix elements.
        */
    typename NormTraits<Matrix>::NormType norm() const;

        /** create a transposed view of this matrix.
            No data are copied. If you want to transpose this matrix permanently,
            you have to assign the transposed view:

            \code
            a = a.transpose();
            \endcode
         */
    ArrayViewND<2, vluae_type, StridedArrayTag> transpose() const;
#endif

        /** add \a other to this (sizes must match).
         */
    template <class U>
    Matrix & operator+=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator+=(other);
        return *this;
    }

        /** subtract \a other from this (sizes must match).
         */
    template <class U>
    Matrix & operator-=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator-=(other);
        return *this;
    }

        /** multiply \a other element-wise with this matrix (sizes must match).
         */
    template <class U>
    Matrix & operator*=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator*=(other);
        return *this;
    }

        /** divide this matrix element-wise by \a other (sizes must match).
         */
    template <class U>
    Matrix & operator/=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator/=(other);
        return *this;
    }

        /** add \a other to each element of this matrix
         */
    Matrix & operator+=(T other)
    {
        BaseType::operator+=(other);
        return *this;
    }

        /** subtract \a other from each element of this matrix
         */
    Matrix & operator-=(T other)
    {
        BaseType::operator-=(other);
        return *this;
    }

        /** scalar multiply this with \a other
         */
    Matrix & operator*=(T other)
    {
        BaseType::operator*=(other);
        return *this;
    }

        /** scalar divide this by \a other
         */
    Matrix & operator/=(T other)
    {
        BaseType::operator/=(other);
        return *this;
    }
};

// TemporaryMatrix is provided as an optimization: Functions returning a matrix can
// use TemporaryMatrix to make explicit that it was allocated as a temporary data structure.
// Functions receiving a TemporaryMatrix can thus often avoid to allocate new temporary
// memory.
template <class T, class ALLOC>  // default ALLOC already declared above
class TemporaryMatrix
: public Matrix<T, ALLOC>
{
    typedef Matrix<T, ALLOC> BaseType;
  public:
    typedef Matrix<T, ALLOC>                        matrix_type;
    typedef TemporaryMatrix<T, ALLOC>               temp_type;
    typedef ArrayViewND<2, T, StridedArrayTag>   view_type;
    typedef typename BaseType::value_type           value_type;
    typedef typename BaseType::pointer              pointer;
    typedef typename BaseType::const_pointer        const_pointer;
    typedef typename BaseType::reference            reference;
    typedef typename BaseType::const_reference      const_reference;
    typedef typename BaseType::difference_type      difference_type;
    typedef typename BaseType::difference_type_1    difference_type_1;
    typedef ALLOC                                   allocator_type;

    TemporaryMatrix(difference_type const & shape)
    : BaseType(shape, ALLOC())
    {}

    TemporaryMatrix(difference_type const & shape, const_reference init)
    : BaseType(shape, init, ALLOC())
    {}

    TemporaryMatrix(difference_type_1 rows, difference_type_1 columns)
    : BaseType(rows, columns, ALLOC())
    {}

    TemporaryMatrix(difference_type_1 rows, difference_type_1 columns, const_reference init)
    : BaseType(rows, columns, init, ALLOC())
    {}

    template<class U>
    TemporaryMatrix(const ArrayViewND<2, U> &rhs)
    : BaseType(rhs)
    {}

    TemporaryMatrix(const TemporaryMatrix &rhs)
    : BaseType()
    {
        this->swap(const_cast<TemporaryMatrix &>(rhs));
    }

    template <class U>
    TemporaryMatrix & init(const U & init)
    {
        BaseType::init(init);
        return *this;
    }

    template <class U>
    TemporaryMatrix & operator+=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator+=(other);
        return *this;
    }

    template <class U>
    TemporaryMatrix & operator-=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator-=(other);
        return *this;
    }

    template <class U>
    TemporaryMatrix & operator*=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator*=(other);
        return *this;
    }

    template <class U>
    TemporaryMatrix & operator/=(ArrayViewND<2, U> const & other)
    {
        BaseType::operator/=(other);
        return *this;
    }

    TemporaryMatrix & operator+=(T other)
    {
        BaseType::operator+=(other);
        return *this;
    }

    TemporaryMatrix & operator-=(T other)
    {
        BaseType::operator-=(other);
        return *this;
    }

    TemporaryMatrix & operator*=(T other)
    {
        BaseType::operator*=(other);
        return *this;
    }

    TemporaryMatrix & operator/=(T other)
    {
        BaseType::operator/=(other);
        return *this;
    }
  private:

    TemporaryMatrix &operator=(const TemporaryMatrix &rhs); // intentionally not implemented
};

#endif // if 0

/** \defgroup LinearAlgebraFunctions Matrix Functions

    \brief Basic matrix algebra, element-wise mathematical functions, row and columns statistics, data normalization etc.

    \ingroup LinearAlgebraModule
 */
//@{

    /** Number of rows of a matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
inline ArrayIndex
rowCount(const ArrayViewND<2, T> &x)
{
    return x.shape(0);
}

/** Number of columns of a matrix.

<b>\#include</b> \<vigra2/matrix.hxx\> or<br>
<b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
Namespaces: vigra and vigra::linalg
*/
template <class T>
inline ArrayIndex
columnCount(const ArrayViewND<2, T> &x)
{
    return x.shape(1);
}

/** Number of elements of a matrix.

<b>\#include</b> \<vigra2/matrix.hxx\> or<br>
<b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
Namespaces: vigra and vigra::linalg
*/
template <class T>
inline ArrayIndex
elementCount(const ArrayViewND<2, T> &x)
{
    return x.size();
}

    /** Create a row vector view for row \a d of the matrix \a m

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
inline ArrayViewND <2, T>
rowVector(ArrayViewND <2, T> const & m, ArrayIndex d)
{
    return m.subarray(Shape<2>(d, 0), Shape<2>(d+1, columnCount(m)));
}


    /** Create a row vector view of the matrix \a m starting at element \a first and ranging
        to column \a end (non-inclusive).

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
inline ArrayViewND <2, T>
rowVector(ArrayViewND <2, T> const & m, Shape<2> const & first, ArrayIndex end)
{
    return m.subarray(first, Shape<2>(first[0]+1, end));
}

    /** Create a column vector view for column \a d of the matrix \a m

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
inline ArrayViewND <2, T>
columnVector(ArrayViewND<2, T> const & m, ArrayIndex d)
{
    return m.subarray(Shape<2>(0, d), Shape<2>(rowCount(m), d+1));
}

    /** Create a column vector view of the matrix \a m starting at element \a first and
        ranging to row \a end (non-inclusive).

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     **/
template <class T>
inline ArrayViewND <2, T>
columnVector(ArrayViewND<2, T> const & m, Shape<2> const & first, int end)
{
    return m.subarray(first, Shape<2>(end, first[1]+1));
}

    /** Create a sub vector view of the vector \a m starting at element \a first and
        ranging to row \a end (non-inclusive).

        Note: This function may only be called when either <tt>rowCount(m) == 1</tt> or
        <tt>columnCount(m) == 1</tt>, i.e. when \a m really represents a vector.
        Otherwise, a PreconditionViolation exception is raised.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     **/
template <class T>
inline ArrayViewND <2, T>
subVector(ArrayViewND<2, T> const & m, int first, int end)
{
    if(columnCount(m) == 1)
        return m.subarray(Shape<2>(first, 0), Shape<2>(end, 1));
    vigra_precondition(rowCount(m) == 1,
                       "linalg::subVector(): Input must be a vector (1xN or Nx1).");
    return m.subarray(Shape<2>(0, first), Shape<2>(1, end));
}

    /** Check whether matrix \a m is symmetric.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
bool
isSymmetric(ArrayViewND<2, T> const & m)
{
    const ArrayIndex size = rowCount(m);
    if(size != columnCount(m))
        return false;

    for(ArrayIndex i = 0; i < size; ++i)
        for(ArrayIndex j = i+1; j < size; ++j)
            if(m(j, i) != m(i, j))
                return false;
    return true;
}


    /** Compute the trace of a square matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
typename NumericTraits<T>::Promote
trace(ArrayViewND<2, T> const & m)
{
    typedef typename NumericTraits<T>::Promote SumType;

    const ArrayIndex size = rowCount(m);
    vigra_precondition(size == columnCount(m), "linalg::trace(): Matrix must be square.");

    SumType sum = NumericTraits<SumType>::zero();
    for(ArrayIndex i = 0; i < size; ++i)
        sum += m(i, i);
    return sum;
}

    /** initialize the given square matrix as an identity matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
void identityMatrix(ArrayViewND<2, T> &r)
{
    const ArrayIndex rows = rowCount(r);
    vigra_precondition(rows == columnCount(r),
       "identityMatrix(): Matrix must be square.");
    for(ArrayIndex i = 0; i < rows; ++i) {
        for(ArrayIndex j = 0; j < rows; ++j)
            r(j, i) = NumericTraits<T>::zero();
        r(i, i) = NumericTraits<T>::one();
    }
}

    /** create an identity matrix of the given size.
        Usage:

        \code
        vigra::Matrix<double> m = vigra::identityMatrix<double>(size);
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T>
identityMatrix(ArrayIndex size)
{
    Matrix<T> ret(size, size);
    for(ArrayIndex i = 0; i < size; ++i)
        ret(i, i) = NumericTraits<T>::one();
    return ret;
}

    /** create an identity matrix of the given size.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T>
eye(ArrayIndex size)
{
    return identityMatrix(size);
}

    /** create matrix of ones of the given size.
        Usage:

        \code
        vigra::Matrix<double> m = vigra::ones<double>(rows, cols);
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T>
ones(ArrayIndex rows, ArrayIndex cols)
{
    return Matrix<T>({ rows, cols }, NumericTraits<T>::one());
}



template <class T>
void diagonalMatrixImpl(ArrayViewND<1, T> const & v, ArrayViewND<2, T> &r)
{
    const ArrayIndex size = v.size();
    vigra_precondition(rowCount(r) == size && columnCount(r) == size,
        "diagonalMatrix(): result must be a square matrix.");
    for(ArrayIndex i = 0; i < size; ++i)
        r(i, i) = v(i);
}

    /** make a diagonal matrix from a vector.
        The vector is given as matrix \a v, which must either have a single
        row or column. The result is written into the square matrix \a r.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
void diagonalMatrix(ArrayViewND<2, T> const & v, ArrayViewND<2, T> &r)
{
    vigra_precondition(rowCount(v) == 1 || columnCount(v) == 1,
        "diagonalMatrix(): input must be a vector.");
    r = T();
    if(rowCount(v) == 1)
        diagonalMatrixImpl(v.bind(0, 0), r);
    else
        diagonalMatrixImpl(v.bind(1, 0), r);
}

    /** create a diagonal matrix from a vector.
        The vector is given as matrix \a v, which must either have a single
        row or column. The result is returned as a temporary matrix.
        Usage:

        \code
        vigra::Matrix<double> v(1, len);
        v = ...;

        vigra::Matrix<double> m = diagonalMatrix(v);
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T> diagonalMatrix(ArrayViewND<2, T> const & v)
{
    vigra_precondition(rowCount(v) == 1 || columnCount(v) == 1,
        "diagonalMatrix(): input must be a vector.");
    ArrayIndex size = v.size();
    Matrix<T> ret(size, size);
    if(rowCount(v) == 1)
        diagonalMatrixImpl(v.bind(0, 0), ret);
    else
        diagonalMatrixImpl(v.bind(1, 0), ret);
    return ret;
}

using vigra::transpose;

    /** transpose matrix \a v.
        The result is written into \a r which must have the correct (i.e.
        transposed) shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
void transpose(const ArrayViewND<2, T> &v, ArrayViewND<2, T> &r)
{
    const ArrayIndex rows = rowCount(r);
    const ArrayIndex cols = columnCount(r);
    vigra_precondition(rows == columnCount(v) && cols == rowCount(v),
       "transpose(): arrays must have transposed shapes.");
    for(ArrayIndex i = 0; i < cols; ++i)
        for(ArrayIndex j = 0; j < rows; ++j)
            r(j, i) = v(i, j);
}

    /** Create new matrix by concatenating two matrices \a a and \a b vertically, i.e. on top of each other.
        The two matrices must have the same number of columns.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
joinVertically(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    ArrayIndex n = columnCount(a);
    vigra_precondition(n == columnCount(b),
       "joinVertically(): shape mismatch.");

    ArrayIndex ma = rowCount(a);
    ArrayIndex mb = rowCount(b);
    Matrix<T> t(ma + mb, n, T());
    t.subarray(Shape<2>(0,0), Shape<2>(ma, n)) = a;
    t.subarray(Shape<2>(ma,0), Shape<2>(ma+mb, n)) = b;
    return t;
}

    /** Create new matrix by concatenating two matrices \a a and \a b horizontally, i.e. side by side.
        The two matrices must have the same number of rows.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
joinHorizontally(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    ArrayIndex m = rowCount(a);
    vigra_precondition(m == rowCount(b),
       "joinHorizontally(): shape mismatch.");

    ArrayIndex na = columnCount(a);
    ArrayIndex nb = columnCount(b);
    Matrix<T> t(m, na + nb, T());
    t.subarray(Shape<2>(0,0), Shape<2>(m, na)) = a;
    t.subarray(Shape<2>(0, na), Shape<2>(m, na + nb)) = b;
    return t;
}

    /** Initialize a matrix with repeated copies of a given matrix.

        Matrix \a r will consist of \a verticalCount downward repetitions of \a v,
        and \a horizontalCount side-by-side repetitions. When \a v has size <tt>m</tt> by <tt>n</tt>,
        \a r must have size <tt>(m*verticalCount)</tt> by <tt>(n*horizontalCount)</tt>.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void repeatMatrix(ArrayViewND<2, T> const & v, ArrayViewND<2, T> &r,
                  unsigned int verticalCount, unsigned int horizontalCount)
{
    ArrayIndex m = rowCount(v), n = columnCount(v);
    vigra_precondition(m*verticalCount == rowCount(r) && n*horizontalCount == columnCount(r),
        "repeatMatrix(): Shape mismatch.");

    for(ArrayIndex l=0; l<static_cast<ArrayIndex>(horizontalCount); ++l)
    {
        for(ArrayIndex k=0; k<static_cast<ArrayIndex>(verticalCount); ++k)
        {
            r.subarray(Shape<2>(k*m, l*n), Shape<2>((k+1)*m, (l+1)*n)) = v;
        }
    }
}

    /** Create a new matrix by repeating a given matrix.

        The resulting matrix \a r will consist of \a verticalCount downward repetitions of \a v,
        and \a horizontalCount side-by-side repetitions, i.e. it will be of size
        <tt>(m*verticalCount)</tt> by <tt>(n*horizontalCount)</tt> when \a v has size <tt>m</tt> by <tt>n</tt>.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
Matrix<T>
repeatMatrix(ArrayViewND<2, T> const & v, unsigned int verticalCount, unsigned int horizontalCount)
{
    ArrayIndex m = rowCount(v), n = columnCount(v);
    Matrix<T> ret(verticalCount*m, horizontalCount*n);
    repeatMatrix(v, ret, verticalCount, horizontalCount);
    return ret;
}

    /** add matrices \a a and \a b.
        The result is written into \a r. All three matrices must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void add(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b,
              ArrayViewND<2, T> &r)
{
    const ArrayIndex rrows = rowCount(r);
    const ArrayIndex rcols = columnCount(r);
    vigra_precondition(rrows == rowCount(a) && rcols == columnCount(a) &&
                       rrows == rowCount(b) && rcols == columnCount(b),
                       "add(): Matrix shapes must agree.");

    for(ArrayIndex i = 0; i < rcols; ++i) {
        for(ArrayIndex j = 0; j < rrows; ++j) {
            r(j, i) = a(j, i) + b(j, i);
        }
    }
}

    /** add matrices \a a and \a b.
        The two matrices must have the same shape.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator+(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    return Matrix<T>(a) += b;
}

template <class T>
inline Matrix<T>
operator+(Matrix<T> && a, const ArrayViewND<2, T> &b)
{
    a += b;
    return std::move(a);
}

template <class T>
inline Matrix<T>
operator+(const ArrayViewND<2, T> &a, Matrix<T> && b)
{
    b += a;
    return std::move(b);
}

template <class T>
inline Matrix<T>
operator+(Matrix<T> && a, Matrix<T> && b)
{
    a += b;
    return std::move(a);
}

    /** add scalar \a b to every element of the matrix \a a.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator+(const ArrayViewND<2, T> &a, T b)
{
    return Matrix<T>(a) += b;
}

template <class T>
inline Matrix<T>
operator+(Matrix<T> && a, T b)
{
    a += b;
    return std::move(a);
}

    /** add scalar \a a to every element of the matrix \a b.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator+(T a, const ArrayViewND<2, T> &b)
{
    return Matrix<T>(b) += a;
}

template <class T>
inline Matrix<T>
operator+(T a, Matrix<T> && b)
{
    b += a;
    return std::move(b);
}

    /** subtract matrix \a b from \a a.
        The result is written into \a r. All three matrices must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void sub(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b,
         ArrayViewND<2, T> &r)
{
    const ArrayIndex rrows = rowCount(r);
    const ArrayIndex rcols = columnCount(r);
    vigra_precondition(rrows == rowCount(a) && rcols == columnCount(a) &&
                       rrows == rowCount(b) && rcols == columnCount(b),
                       "subtract(): Matrix shapes must agree.");

    for(ArrayIndex i = 0; i < rcols; ++i) {
        for(ArrayIndex j = 0; j < rrows; ++j) {
            r(j, i) = a(j, i) - b(j, i);
        }
    }
}

    /** subtract matrix \a b from \a a.
        The two matrices must have the same shape.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator-(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    return Matrix<T>(a) -= b;
}

template <class T>
inline Matrix<T>
operator-(Matrix<T> && a, const ArrayViewND<2, T> &b)
{
    a -= b;
    return std::move(a);
}

template <class T>
Matrix<T>
operator-(const ArrayViewND<2, T> &a, Matrix<T> && b)
{
    const ArrayIndex rows = rowCount(a);
    const ArrayIndex cols = columnCount(a);
    vigra_precondition(rows == b.rowCount() && cols == b.columnCount(),
       "Matrix::operator-(): Shape mismatch.");

    for(ArrayIndex i = 0; i < cols; ++i)
        for(ArrayIndex j = 0; j < rows; ++j)
            b(j, i) = a(j, i) - b(j, i);
    return std::move(b);
}

template <class T>
inline Matrix<T>
operator-(Matrix<T> && a, Matrix<T> && b)
{
    a -= b;
    return std::move(a);
}

    /** negate matrix \a a.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator-(const ArrayViewND<2, T> &a)
{
    return Matrix<T>(a) *= -NumericTraits<T>::one();
}

template <class T>
inline Matrix<T>
operator-(Matrix<T> && a)
{
    a *= -NumericTraits<T>::one();
    return std::move(a);
}

    /** subtract scalar \a b from every element of the matrix \a a.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator-(const ArrayViewND<2, T> &a, T b)
{
    return Matrix<T>(a) -= b;
}

template <class T>
inline Matrix<T>
operator-(Matrix<T> && a, T b)
{
    a -= b;
    return std::move(a);
}

    /** subtract every element of the matrix \a b from scalar \a a.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator-(T a, const ArrayViewND<2, T> &b)
{
    return Matrix<T>(b.shape(), a) -= b;
}

    /** calculate the inner product of two matrices representing vectors.
        Typically, matrix \a x has a single row, and matrix \a y has
        a single column, and the other dimensions match. In addition, this
        function handles the cases when either or both of the two inputs are
        transposed (e.g. it can compute the dot product of two column vectors).
        A <tt>PreconditionViolation</tt> exception is thrown when
        the shape conditions are violated.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
typename NormTraits<T>::SquaredNormType
dot(const ArrayViewND<2, T> &x, const ArrayViewND<2, T> &y)
{
    typename NormTraits<T>::SquaredNormType ret =
           NumericTraits<typename NormTraits<T>::SquaredNormType>::zero();
    if(y.shape(1) == 1)
    {
        std::ptrdiff_t size = y.shape(0);
        if(x.shape(0) == 1 && x.shape(1) == size) // proper scalar product
            for(std::ptrdiff_t i = 0; i < size; ++i)
                ret += x(0, i) * y(i, 0);
        else if(x.shape(1) == 1u && x.shape(0) == size) // two column vectors
            for(std::ptrdiff_t i = 0; i < size; ++i)
                ret += x(i, 0) * y(i, 0);
        else
            vigra_precondition(false, "dot(): wrong matrix shapes.");
    }
    else if(y.shape(0) == 1)
    {
        std::ptrdiff_t size = y.shape(1);
        if(x.shape(0) == 1u && x.shape(1) == size) // two row vectors
            for(std::ptrdiff_t i = 0; i < size; ++i)
                ret += x(0, i) * y(0, i);
        else if(x.shape(1) == 1u && x.shape(0) == size) // column dot row
            for(std::ptrdiff_t i = 0; i < size; ++i)
                ret += x(i, 0) * y(0, i);
        else
            vigra_precondition(false, "dot(): wrong matrix shapes.");
    }
    else
            vigra_precondition(false, "dot(): wrong matrix shapes.");
    return ret;
}

    /** calculate the inner product of two vectors. The vector
        lengths must match.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
typename NormTraits<T>::SquaredNormType
dot(const ArrayViewND<1, T> &x, const ArrayViewND<1, T> &y)
{
    const ArrayIndex n = x.elementCount();
    vigra_precondition(n == y.elementCount(),
       "dot(): shape mismatch.");
    typename NormTraits<T>::SquaredNormType ret =
                NumericTraits<typename NormTraits<T>::SquaredNormType>::zero();
    for(ArrayIndex i = 0; i < n; ++i)
        ret += x(i) * y(i);
    return ret;
}

    /** calculate the cross product of two vectors of length 3.
        The result is written into \a r.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
void cross(const ArrayViewND<1, T> &x, const ArrayViewND<1, T> &y,
           ArrayViewND<1, T> &r)
{
    vigra_precondition(3 == x.elementCount() && 3 == y.elementCount() && 3 == r.elementCount(),
       "cross(): vectors must have length 3.");
    r(0) = x(1)*y(2) - x(2)*y(1);
    r(1) = x(2)*y(0) - x(0)*y(2);
    r(2) = x(0)*y(1) - x(1)*y(0);
}

    /** calculate the cross product of two matrices representing vectors.
        That is, \a x, \a y, and \a r must have a single column of length 3. The result
        is written into \a r.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
void cross(const ArrayViewND<2, T> &x, const ArrayViewND<2, T> &y,
           ArrayViewND<2, T> &r)
{
    vigra_precondition(3 == rowCount(x) && 3 == rowCount(y) && 3 == rowCount(r) ,
       "cross(): vectors must have length 3.");
    r(0,0) = x(1,0)*y(2,0) - x(2,0)*y(1,0);
    r(1,0) = x(2,0)*y(0,0) - x(0,0)*y(2,0);
    r(2,0) = x(0,0)*y(1,0) - x(1,0)*y(0,0);
}

    /** calculate the cross product of two matrices representing vectors.
        That is, \a x, and \a y must have a single column of length 3. The result
        is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T>
cross(const ArrayViewND<2, T> &x, const ArrayViewND<2, T> &y)
{
    Matrix<T> ret(3, 1);
    cross(x, y, ret);
    return ret;
}
    /** calculate the outer product of two matrices representing vectors.
        That is, matrix \a x must have a single column, and matrix \a y must
        have a single row, and the other dimensions must match. The result
        is written into \a r.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
void outer(const ArrayViewND<2, T> &x, const ArrayViewND<2, T> &y,
      ArrayViewND<2, T> &r)
{
    const ArrayIndex rows = rowCount(r);
    const ArrayIndex cols = columnCount(r);
    vigra_precondition(rows == rowCount(x) && cols == columnCount(y) &&
                       1 == columnCount(x) && 1 == rowCount(y),
       "outer(): shape mismatch.");
    for(ArrayIndex i = 0; i < cols; ++i)
        for(ArrayIndex j = 0; j < rows; ++j)
            r(j, i) = x(j, 0) * y(0, i);
}

    /** calculate the outer product of two matrices representing vectors.
        That is, matrix \a x must have a single column, and matrix \a y must
        have a single row, and the other dimensions must match. The result
        is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T>
outer(const ArrayViewND<2, T> &x, const ArrayViewND<2, T> &y)
{
    const ArrayIndex rows = rowCount(x);
    const ArrayIndex cols = columnCount(y);
    vigra_precondition(1 == columnCount(x) && 1 == rowCount(y),
       "outer(): shape mismatch.");
    Matrix<T> ret(rows, cols);
    outer(x, y, ret);
    return ret;
}

    /** calculate the outer product of a matrix (representing a vector) with itself.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T>
Matrix<T>
outer(const ArrayViewND<2, T> &x)
{
    const ArrayIndex rows = rowCount(x);
    const ArrayIndex cols = columnCount(x);
    vigra_precondition(rows == 1 || cols == 1,
       "outer(): matrix does not represent a vector.");
    const ArrayIndex size = std::max(rows, cols);
    Matrix<T> ret(size, size);

    if(rows == 1)
    {
        for(ArrayIndex i = 0; i < size; ++i)
            for(ArrayIndex j = 0; j < size; ++j)
                ret(j, i) = x(0, j) * x(0, i);
    }
    else
    {
        for(ArrayIndex i = 0; i < size; ++i)
            for(ArrayIndex j = 0; j < size; ++j)
                ret(j, i) = x(j, 0) * x(i, 0);
    }
    return ret;
}

    /** calculate the outer product of a TinyArray with itself.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespaces: vigra and vigra::linalg
     */
template <class T, int N>
Matrix<T>
outer(const TinyArray<T, N> &x)
{
    Matrix<T> ret(N, N);

    for(ArrayIndex i = 0; i < N; ++i)
        for(ArrayIndex j = 0; j < N; ++j)
            ret(j, i) = x[j] * x[i];
    return ret;
}

template <class T>
class PointWise
{
  public:
    T const & t;

    PointWise(T const & it)
    : t(it)
    {}
};

template <class T>
PointWise<T> pointWise(T const & t)
{
    return PointWise<T>(t);
}


    /** multiply matrix \a a with scalar \a b.
        The result is written into \a r. \a a and \a r must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void smul(const ArrayViewND<2, T> &a, T b, ArrayViewND<2, T> &r)
{
    const ArrayIndex rows = rowCount(a);
    const ArrayIndex cols = columnCount(a);
    vigra_precondition(rows == rowCount(r) && cols == columnCount(r),
                       "smul(): Matrix sizes must agree.");

    for(ArrayIndex i = 0; i < cols; ++i)
        for(ArrayIndex j = 0; j < rows; ++j)
            r(j, i) = a(j, i) * b;
}

    /** multiply scalar \a a with matrix \a b.
        The result is written into \a r. \a b and \a r must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void smul(T a, const ArrayViewND<2, T> &b, ArrayViewND<2, T> &r)
{
    smul(b, a, r);
}

    /** perform matrix multiplication of matrices \a a and \a b.
        The result is written into \a r. The three matrices must have matching shapes.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void mmul(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b,
         ArrayViewND<2, T> &r)
{
    const ArrayIndex rrows = rowCount(r);
    const ArrayIndex rcols = columnCount(r);
    const ArrayIndex acols = columnCount(a);
    vigra_precondition(rrows == rowCount(a) && rcols == columnCount(b) && acols == rowCount(b),
                       "mmul(): Matrix shapes must agree.");

    // order of loops ensures that inner loop goes down columns
    for(ArrayIndex i = 0; i < rcols; ++i)
    {
        for(ArrayIndex j = 0; j < rrows; ++j)
            r(j, i) = a(j, 0) * b(0, i);
        for(ArrayIndex k = 1; k < acols; ++k)
            for(ArrayIndex j = 0; j < rrows; ++j)
                r(j, i) += a(j, k) * b(k, i);
    }
}

    /** perform matrix multiplication of matrices \a a and \a b.
        \a a and \a b must have matching shapes.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
mmul(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    Matrix<T> ret(rowCount(a), columnCount(b));
    mmul(a, b, ret);
    return ret;
}

    /** multiply two matrices \a a and \a b pointwise.
        The result is written into \a r. All three matrices must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void pmul(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b,
              ArrayViewND<2, T> &r)
{
    const ArrayIndex rrows = rowCount(r);
    const ArrayIndex rcols = columnCount(r);
    vigra_precondition(rrows == rowCount(a) && rcols == columnCount(a) &&
                       rrows == rowCount(b) && rcols == columnCount(b),
                       "pmul(): Matrix shapes must agree.");

    for(ArrayIndex i = 0; i < rcols; ++i) {
        for(ArrayIndex j = 0; j < rrows; ++j) {
            r(j, i) = a(j, i) * b(j, i);
        }
    }
}

    /** multiply matrices \a a and \a b pointwise.
        \a a and \a b must have matching shapes.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
pmul(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    Matrix<T> ret(a.shape());
    pmul(a, b, ret);
    return ret;
}

    /** multiply matrices \a a and \a b pointwise.
        \a a and \a b must have matching shapes.
        The result is returned as a temporary matrix.

        Usage:

        \code
        Matrix<double> a(m,n), b(m,n);

        Matrix<double> c = a * pointWise(b);
        // is equivalent to
        // Matrix<double> c = pmul(a, b);
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T, class U>
inline Matrix<T>
operator*(const ArrayViewND<2, T> &a, PointWise<U> b)
{
    return pmul(a, b.t);
}

    /** multiply matrix \a a with scalar \a b.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator*(const ArrayViewND<2, T> &a, T b)
{
    return Matrix<T>(a) *= b;
}

template <class T>
inline Matrix<T>
operator*(Matrix<T> && a, T b)
{
    a *= b;
    return std::move(a);
}

    /** multiply scalar \a a with matrix \a b.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator*(T a, const ArrayViewND<2, T> &b)
{
    return Matrix<T>(b) *= a;
}

template <class T>
inline Matrix<T>
operator*(T a, Matrix<T> && b)
{
    b *= a;
    return std::move(b);
}

    /** multiply matrix \a a with TinyArray \a b.
        \a a must be of size <tt>N x N</tt>. Vector \a b and the result
        vector are interpreted as column vectors.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T, class DERIVED, int N>
TinyArray<T, N>
operator*(const Matrix<T> &a, const TinyArrayBase<T, DERIVED, N> &b)
{
    vigra_precondition(N == rowCount(a) && N == columnCount(a),
         "operator*(Matrix, TinyArray): Shape mismatch.");

    TinyArray<T, N> res;
    for (ArrayIndex i = 0; i < N; ++i)
        for (ArrayIndex j = 0; j < N; ++j)
            res[i] += a(i,j) * b[j];
    return res;
}

    /** multiply TinyArray \a a with matrix \a b.
        \a b must be of size <tt>N x N</tt>. Vector \a a and the result
        vector are interpreted as row vectors.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T, class DERIVED, int N>
TinyArray<T, N>
operator*(const TinyArrayBase<T, DERIVED, N> &a, const Matrix<T> &b)
{
    vigra_precondition(N == rowCount(b) && N == columnCount(b),
         "operator*(TinyArray, Matrix): Shape mismatch.");

    TinyArray<T, N> res;
    for (ArrayIndex i = 0; i < N; ++i)
        for (ArrayIndex j = 0; j < N; ++j)
            res[j] += a[i] * b(i, j);
    return res;
}

    /** perform matrix multiplication of matrices \a a and \a b.
        \a a and \a b must have matching shapes.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator*(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    Matrix<T> ret(rowCount(a), columnCount(b));
    mmul(a, b, ret);
    return ret;
}

    /** divide matrix \a a by scalar \a b.
        The result is written into \a r. \a a and \a r must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void sdiv(const ArrayViewND<2, T> &a, T b, ArrayViewND<2, T> &r)
{
    const ArrayIndex rows = rowCount(a);
    const ArrayIndex cols = columnCount(a);
    vigra_precondition(rows == rowCount(r) && cols == columnCount(r),
                       "sdiv(): Matrix sizes must agree.");

    for(ArrayIndex i = 0; i < cols; ++i)
        for(ArrayIndex j = 0; j < rows; ++j)
            r(j, i) = a(j, i) / b;
}

    /** divide two matrices \a a and \a b pointwise.
        The result is written into \a r. All three matrices must have the same shape.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
void pdiv(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b,
              ArrayViewND<2, T> &r)
{
    const ArrayIndex rrows = rowCount(r);
    const ArrayIndex rcols = columnCount(r);
    vigra_precondition(rrows == rowCount(a) && rcols == columnCount(a) &&
                       rrows == rowCount(b) && rcols == columnCount(b),
                       "pdiv(): Matrix shapes must agree.");

    for(ArrayIndex i = 0; i < rcols; ++i) {
        for(ArrayIndex j = 0; j < rrows; ++j) {
            r(j, i) = a(j, i) / b(j, i);
        }
    }
}

    /** divide matrices \a a and \a b pointwise.
        \a a and \a b must have matching shapes.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
pdiv(const ArrayViewND<2, T> &a, const ArrayViewND<2, T> &b)
{
    Matrix<T> ret(a.shape());
    pdiv(a, b, ret);
    return ret;
}

    /** divide matrices \a a and \a b pointwise.
        \a a and \a b must have matching shapes.
        The result is returned as a temporary matrix.

        Usage:

        \code
        Matrix<double> a(m,n), b(m,n);

        Matrix<double> c = a / pointWise(b);
        // is equivalent to
        // Matrix<double> c = pdiv(a, b);
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T, class U>
inline Matrix<T>
operator/(const ArrayViewND<2, T> &a, PointWise<U> b)
{
    return pdiv(a, b.t);
}

    /** divide matrix \a a by scalar \a b.
        The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator/(const ArrayViewND<2, T> &a, T b)
{
    return Matrix<T>(a) /= b;
}

template <class T>
inline Matrix<T>
operator/(Matrix<T> && a, T b)
{
    a /= b;
    return std::move(a);
}

    /** Create a matrix whose elements are the quotients between scalar \a a and
        matrix \a b. The result is returned as a temporary matrix.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: vigra::linalg
     */
template <class T>
inline Matrix<T>
operator/(T a, const ArrayViewND<2, T> &b)
{
    return Matrix<T>(b.shape(), a) / pointWise(b);
}

using vigra::argMin;
using vigra::argMinIf;
using vigra::argMax;
using vigra::argMaxIf;

    /** \brief Find the index of the minimum element in a matrix.

        The function returns the index in column-major scan-order sense,
        i.e. according to the order used by <tt>ArrayViewND::operator[]</tt>.
        If the matrix is actually a vector, this is just the row or columns index.
        In case of a truly 2-dimensional matrix, the index can be converted to an
        index pair by calling <tt>ArrayViewND::scanOrderIndexToCoordinate()</tt>

        <b>Required Interface:</b>

        \code
        bool f = a[0] < NumericTraits<T>::max();
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T>
int argMin(ArrayViewND<2, T> const & a)
{
    T vopt = NumericTraits<T>::max();
    int best = -1;
    for(int k=0; k < a.size(); ++k)
    {
        if(a[k] < vopt)
        {
            vopt = a[k];
            best = k;
        }
    }
    return best;
}

    /** \brief Find the index of the maximum element in a matrix.

        The function returns the index in column-major scan-order sense,
        i.e. according to the order used by <tt>ArrayViewND::operator[]</tt>.
        If the matrix is actually a vector, this is just the row or columns index.
        In case of a truly 2-dimensional matrix, the index can be converted to an
        index pair by calling <tt>ArrayViewND::scanOrderIndexToCoordinate()</tt>

        <b>Required Interface:</b>

        \code
        bool f = NumericTraits<T>::min() < a[0];
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T>
int argMax(ArrayViewND<2, T> const & a)
{
    T vopt = NumericTraits<T>::min();
    int best = -1;
    for(int k=0; k < a.size(); ++k)
    {
        if(vopt < a[k])
        {
            vopt = a[k];
            best = k;
        }
    }
    return best;
}

    /** \brief Find the index of the minimum element in a matrix subject to a condition.

        The function returns <tt>-1</tt> if no element conforms to \a condition.
        Otherwise, the index of the maximum element is returned in column-major scan-order sense,
        i.e. according to the order used by <tt>ArrayViewND::operator[]</tt>.
        If the matrix is actually a vector, this is just the row or columns index.
        In case of a truly 2-dimensional matrix, the index can be converted to an
        index pair by calling <tt>ArrayViewND::scanOrderIndexToCoordinate()</tt>

        <b>Required Interface:</b>

        \code
        bool c = condition(a[0]);
        bool f = a[0] < NumericTraits<T>::max();
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T, class UnaryFunctor>
int argMinIf(ArrayViewND<2, T> const & a, UnaryFunctor condition)
{
    T vopt = NumericTraits<T>::max();
    int best = -1;
    for(int k=0; k < a.size(); ++k)
    {
        if(condition(a[k]) && a[k] < vopt)
        {
            vopt = a[k];
            best = k;
        }
    }
    return best;
}

    /** \brief Find the index of the maximum element in a matrix subject to a condition.

        The function returns <tt>-1</tt> if no element conforms to \a condition.
        Otherwise, the index of the maximum element is returned in column-major scan-order sense,
        i.e. according to the order used by <tt>ArrayViewND::operator[]</tt>.
        If the matrix is actually a vector, this is just the row or columns index.
        In case of a truly 2-dimensional matrix, the index can be converted to an
        index pair by calling <tt>ArrayViewND::scanOrderIndexToCoordinate()</tt>

        <b>Required Interface:</b>

        \code
        bool c = condition(a[0]);
        bool f = NumericTraits<T>::min() < a[0];
        \endcode

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T, class UnaryFunctor>
int argMaxIf(ArrayViewND<2, T> const & a, UnaryFunctor condition)
{
    T vopt = NumericTraits<T>::min();
    int best = -1;
    for(int k=0; k < a.size(); ++k)
    {
        if(condition(a[k]) && vopt < a[k])
        {
            vopt = a[k];
            best = k;
        }
    }
    return best;
}

/** Matrix point-wise power.
*/
template <class T>
linalg::Matrix<T>
pow(ArrayViewND<2, T> const & v, T exponent)
{
    linalg::Matrix<T> t(v.shape());
    ArrayIndex m = rowCount(v), n = columnCount(v);

    for(ArrayIndex i = 0; i < n; ++i)
        for(ArrayIndex j = 0; j < m; ++j)
            t(j, i) = vigra::pow(v(j, i), exponent);
    return t;
}

template <class T>
linalg::Matrix<T>
pow(linalg::Matrix<T> && t, T exponent)
{
    ArrayIndex m = rowCount(t), n = columnCount(t);

    for(ArrayIndex i = 0; i < n; ++i)
        for(ArrayIndex j = 0; j < m; ++j)
            t(j, i) = vigra::pow(t(j, i), exponent);
    return std::move(t);
}

template <class T>
linalg::Matrix<T>
pow(ArrayViewND<2, T> const & v, int exponent)
{
    linalg::Matrix<T> t(v.shape());
    ArrayIndex m = rowCount(v), n = columnCount(v);

    for(ArrayIndex i = 0; i < n; ++i)
        for(ArrayIndex j = 0; j < m; ++j)
            t(j, i) = vigra::pow(v(j, i), exponent);
    return t;
}

template <class T>
linalg::Matrix<T>
pow(linalg::Matrix<T> && t, int exponent)
{
    ArrayIndex m = rowCount(t), n = columnCount(t);

    for(ArrayIndex i = 0; i < n; ++i)
        for(ArrayIndex j = 0; j < m; ++j)
            t(j, i) = vigra::pow(t(j, i), exponent);
    return std::move(t);
}

template <class C>
linalg::Matrix<int>
pow(ArrayViewND<2, int> const & v, int exponent)
{
    linalg::Matrix<int> t(v.shape());
    ArrayIndex m = rowCount(v), n = columnCount(v);

    for(ArrayIndex i = 0; i < n; ++i)
        for(ArrayIndex j = 0; j < m; ++j)
            t(j, i) = static_cast<int>(vigra::pow(static_cast<double>(v(j, i)), exponent));
    return t;
}

inline
linalg::Matrix<int>
pow(linalg::Matrix<int> && t, int exponent)
{
    ArrayIndex m = rowCount(t), n = columnCount(t);

    for(ArrayIndex i = 0; i < n; ++i)
        for(ArrayIndex j = 0; j < m; ++j)
            t(j, i) = static_cast<int>(vigra::pow(static_cast<double>(t(j, i)), exponent));
    return std::move(t);
}

    /** Matrix point-wise sqrt. */
template <class T>
linalg::Matrix<T> sqrt(ArrayViewND<2, T> const & v);
    /** Matrix point-wise exp. */
template <class T>
linalg::Matrix<T> exp(ArrayViewND<2, T> const & v);
    /** Matrix point-wise log. */
template <class T>
linalg::Matrix<T> log(ArrayViewND<2, T> const & v);
    /** Matrix point-wise log10. */
template <class T>
linalg::Matrix<T> log10(ArrayViewND<2, T> const & v);
    /** Matrix point-wise sin. */
template <class T>
linalg::Matrix<T> sin(ArrayViewND<2, T> const & v);
    /** Matrix point-wise asin. */
template <class T>
linalg::Matrix<T> asin(ArrayViewND<2, T> const & v);
    /** Matrix point-wise cos. */
template <class T>
linalg::Matrix<T> cos(ArrayViewND<2, T> const & v);
    /** Matrix point-wise acos. */
template <class T>
linalg::Matrix<T> acos(ArrayViewND<2, T> const & v);
    /** Matrix point-wise tan. */
template <class T>
linalg::Matrix<T> tan(ArrayViewND<2, T> const & v);
    /** Matrix point-wise atan. */
template <class T>
linalg::Matrix<T> atan(ArrayViewND<2, T> const & v);
    /** Matrix point-wise round. */
template <class T>
linalg::Matrix<T> round(ArrayViewND<2, T> const & v);
    /** Matrix point-wise floor. */
template <class T>
linalg::Matrix<T> floor(ArrayViewND<2, T> const & v);
    /** Matrix point-wise ceil. */
template <class T>
linalg::Matrix<T> ceil(ArrayViewND<2, T> const & v);
    /** Matrix point-wise abs. */
template <class T>
linalg::Matrix<T> abs(ArrayViewND<2, T> const & v);
    /** Matrix point-wise square. */
template <class T>
linalg::Matrix<T> sq(ArrayViewND<2, T> const & v);
    /** Matrix point-wise sign. */
template <class T>
linalg::Matrix<T> sign(ArrayViewND<2, T> const & v);

#define VIGRA_MATRIX_UNARY_FUNCTION(FUNCTION, NAMESPACE) \
using NAMESPACE::FUNCTION; \
template <class T> \
linalg::Matrix<T> FUNCTION(ArrayViewND<2, T> const & v) \
{ \
    linalg::Matrix<T> t(v.shape()); \
    ArrayIndex m = rowCount(v), n = columnCount(v); \
 \
    for(ArrayIndex i = 0; i < n; ++i) \
        for(ArrayIndex j = 0; j < m; ++j) \
            t(j, i) = NAMESPACE::FUNCTION(v(j, i)); \
    return t; \
} \
 \
template <class T> \
linalg::Matrix<T> FUNCTION(linalg::Matrix<T> && t) \
{ \
    ArrayIndex m = rowCount(t), n = columnCount(t); \
 \
    for(ArrayIndex i = 0; i < n; ++i) \
        for(ArrayIndex j = 0; j < m; ++j) \
            t(j, i) = NAMESPACE::FUNCTION(t(j, i)); \
    return std::move(t); \
}\
}\
using linalg::FUNCTION;\
namespace linalg {

VIGRA_MATRIX_UNARY_FUNCTION(sqrt, std)
VIGRA_MATRIX_UNARY_FUNCTION(exp, std)
VIGRA_MATRIX_UNARY_FUNCTION(log, std)
VIGRA_MATRIX_UNARY_FUNCTION(log10, std)
VIGRA_MATRIX_UNARY_FUNCTION(sin, std)
VIGRA_MATRIX_UNARY_FUNCTION(asin, std)
VIGRA_MATRIX_UNARY_FUNCTION(cos, std)
VIGRA_MATRIX_UNARY_FUNCTION(acos, std)
VIGRA_MATRIX_UNARY_FUNCTION(tan, std)
VIGRA_MATRIX_UNARY_FUNCTION(atan, std)
VIGRA_MATRIX_UNARY_FUNCTION(round, vigra)
VIGRA_MATRIX_UNARY_FUNCTION(floor, vigra)
VIGRA_MATRIX_UNARY_FUNCTION(ceil, vigra)
VIGRA_MATRIX_UNARY_FUNCTION(abs, vigra)
VIGRA_MATRIX_UNARY_FUNCTION(sq, vigra)
VIGRA_MATRIX_UNARY_FUNCTION(sign, vigra)

#undef VIGRA_MATRIX_UNARY_FUNCTION

//@}

} // namespace linalg

using linalg::RowMajor;
using linalg::ColumnMajor;
using linalg::Matrix;
using linalg::identityMatrix;
using linalg::eye;
using linalg::ones;
using linalg::diagonalMatrix;
using linalg::transpose;
using linalg::pointWise;
using linalg::trace;
using linalg::dot;
using linalg::cross;
using linalg::outer;
using linalg::rowCount;
using linalg::columnCount;
using linalg::elementCount;
using linalg::rowVector;
using linalg::columnVector;
using linalg::subVector;
using linalg::isSymmetric;
using linalg::joinHorizontally;
using linalg::joinVertically;
using linalg::argMin;
using linalg::argMinIf;
using linalg::argMax;
using linalg::argMaxIf;

} // namespace vigra

namespace std {

/** \addtogroup LinearAlgebraFunctions
 */
//@{

    /** print a matrix \a m to the stream \a s.

        <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
        <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
        Namespace: std
     */
template <class T>
ostream &
operator<<(ostream & s, const vigra::ArrayViewND<2, T> &m)
{
    const vigra::ArrayIndex rows = vigra::linalg::rowCount(m);
    const vigra::ArrayIndex cols = vigra::linalg::columnCount(m);
    ios::fmtflags flags = s.setf(ios::right | ios::fixed, ios::adjustfield | ios::floatfield);
    for(vigra::ArrayIndex j = 0; j < rows; ++j)
    {
        for(vigra::ArrayIndex i = 0; i < cols; ++i)
        {
            s << m(j, i) << " ";
        }
        s << endl;
    }
    s.setf(flags);
    return s;
}

//@}

} // namespace std

namespace vigra {

namespace linalg {

namespace detail {

template <class T1, class T2, class T3>
void
columnStatisticsImpl(ArrayViewND<2, T1> const & A,
                     ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & sumOfSquaredDifferences)
{
    ArrayIndex m = rowCount(A);
    ArrayIndex n = columnCount(A);
    vigra_precondition(1 == rowCount(mean) && n == columnCount(mean) &&
                       1 == rowCount(sumOfSquaredDifferences) && n == columnCount(sumOfSquaredDifferences),
                       "columnStatistics(): Shape mismatch between input and output.");

    // West's algorithm for incremental variance computation
    mean.init(NumericTraits<T2>::zero());
    sumOfSquaredDifferences.init(NumericTraits<T3>::zero());

    for(ArrayIndex k=0; k<m; ++k)
    {
        typedef typename NumericTraits<T2>::RealPromote TmpType;
        Matrix<T2> t = rowVector(A, k) - mean;
        TmpType f  = TmpType(1.0 / (k + 1.0)),
                f1 = TmpType(1.0 - f);
        mean += f*t;
        sumOfSquaredDifferences += f1*sq(t);
    }
}

template <class T1, class T2, class T3>
void
columnStatistics2PassImpl(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & sumOfSquaredDifferences)
{
    ArrayIndex m = rowCount(A);
    ArrayIndex n = columnCount(A);
    vigra_precondition(1 == rowCount(mean) && n == columnCount(mean) &&
                       1 == rowCount(sumOfSquaredDifferences) && n == columnCount(sumOfSquaredDifferences),
                       "columnStatistics(): Shape mismatch between input and output.");

    // two-pass algorithm for incremental variance computation
    mean.init(NumericTraits<T2>::zero());
    for(ArrayIndex k=0; k<m; ++k)
    {
        mean += rowVector(A, k);
    }
    mean /= static_cast<double>(m);

    sumOfSquaredDifferences.init(NumericTraits<T3>::zero());
    for(ArrayIndex k=0; k<m; ++k)
    {
        sumOfSquaredDifferences += sq(rowVector(A, k) - mean);
    }
}

} // namespace detail

/** \addtogroup LinearAlgebraFunctions
 */
//@{
    /** Compute statistics of every column of matrix \a A.

    The result matrices must be row vectors with as many columns as \a A.

    <b> Declarations:</b>

    compute only the mean:
    \code
    namespace vigra { namespace linalg {
        template <class T1, class T2>
        void
        columnStatistics(ArrayViewND<2, T1> const & A,
                         ArrayViewND<2, T2> & mean);
    } }
    \endcode

    compute mean and standard deviation:
    \code
    namespace vigra { namespace linalg {
        template <class T1, class T2, class T3>
        void
        columnStatistics(ArrayViewND<2, T1> const & A,
                         ArrayViewND<2, T2> & mean,
                         ArrayViewND<2, T3> & stdDev);
    } }
    \endcode

    compute mean, standard deviation, and norm:
    \code
    namespace vigra { namespace linalg {
        template <class T1, class T2, class T3, class T4>
        void
        columnStatistics(ArrayViewND<2, T1> const & A,
                         ArrayViewND<2, T2> & mean,
                         ArrayViewND<2, T3> & stdDev,
                         ArrayViewND<2, T4> & norm);
    } }
    \endcode

    <b> Usage:</b>

    <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
    <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
    Namespaces: vigra and vigra::linalg

    \code
    Matrix A(rows, columns);
    .. // fill A
    Matrix columnMean(1, columns), columnStdDev(1, columns), columnNorm(1, columns);

    columnStatistics(A, columnMean, columnStdDev, columnNorm);

    \endcode
    */
doxygen_overloaded_function(template <...> void columnStatistics)

template <class T1, class T2>
void
columnStatistics(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean)
{
    ArrayIndex m = rowCount(A);
    ArrayIndex n = columnCount(A);
    vigra_precondition(1 == rowCount(mean) && n == columnCount(mean),
                       "columnStatistics(): Shape mismatch between input and output.");

    mean.init(NumericTraits<T2>::zero());

    for(ArrayIndex k=0; k<m; ++k)
    {
        mean += rowVector(A, k);
    }
    mean /= T2(m);
}

template <class T1, class T2, class T3>
void
columnStatistics(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & stdDev)
{
    detail::columnStatisticsImpl(A, mean, stdDev);

    if(rowCount(A) > 1)
        stdDev = sqrt(stdDev / T3(rowCount(A) - 1.0));
}

template <class T1, class T2, class T3, class T4>
void
columnStatistics(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & stdDev, ArrayViewND<2, T4> & norm)
{
    ArrayIndex m = rowCount(A);
    ArrayIndex n = columnCount(A);
    vigra_precondition(1 == rowCount(mean) && n == columnCount(mean) &&
                       1 == rowCount(stdDev) && n == columnCount(stdDev) &&
                       1 == rowCount(norm) && n == columnCount(norm),
                       "columnStatistics(): Shape mismatch between input and output.");

    detail::columnStatisticsImpl(A, mean, stdDev);
    norm = sqrt(stdDev + T2(m) * sq(mean));
    stdDev = sqrt(stdDev / T3(m - 1.0));
}

    /** Compute statistics of every row of matrix \a A.

    The result matrices must be column vectors with as many rows as \a A.

    <b> Declarations:</b>

    compute only the mean:
    \code
    namespace vigra { namespace linalg {
        template <class T1, class T2>
        void
        rowStatistics(ArrayViewND<2, T1> const & A,
                      ArrayViewND<2, T2> & mean);
    } }
    \endcode

    compute mean and standard deviation:
    \code
    namespace vigra { namespace linalg {
        template <class T1, class T2, class T3>
        void
        rowStatistics(ArrayViewND<2, T1> const & A,
                      ArrayViewND<2, T2> & mean,
                      ArrayViewND<2, T3> & stdDev);
    } }
    \endcode

    compute mean, standard deviation, and norm:
    \code
    namespace vigra { namespace linalg {
        template <class T1, class T2, class T3, class T4>
        void
        rowStatistics(ArrayViewND<2, T1> const & A,
                      ArrayViewND<2, T2> & mean,
                      ArrayViewND<2, T3> & stdDev,
                      ArrayViewND<2, T4> & norm);
    } }
    \endcode

    <b> Usage:</b>

    <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
    <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
    Namespaces: vigra and vigra::linalg

    \code
    Matrix A(rows, columns);
    .. // fill A
    Matrix rowMean(rows, 1), rowStdDev(rows, 1), rowNorm(rows, 1);

    rowStatistics(a, rowMean, rowStdDev, rowNorm);

    \endcode
     */
doxygen_overloaded_function(template <...> void rowStatistics)

template <class T1, class T2>
void
rowStatistics(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean)
{
    vigra_precondition(1 == columnCount(mean) && rowCount(A) == rowCount(mean),
                       "rowStatistics(): Shape mismatch between input and output.");
    ArrayViewND<2, T2, StridedArrayTag> tm = transpose(mean);
    columnStatistics(transpose(A), tm);
}

template <class T1, class T2, class T3>
void
rowStatistics(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & stdDev)
{
    vigra_precondition(1 == columnCount(mean) && rowCount(A) == rowCount(mean) &&
                       1 == columnCount(stdDev) && rowCount(A) == rowCount(stdDev),
                       "rowStatistics(): Shape mismatch between input and output.");
    ArrayViewND<2, T2, StridedArrayTag> tm = transpose(mean);
    ArrayViewND<2, T3, StridedArrayTag> ts = transpose(stdDev);
    columnStatistics(transpose(A), tm, ts);
}

template <class T1, class T2, class T3, class T4>
void
rowStatistics(ArrayViewND<2, T1> const & A,
                 ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & stdDev, ArrayViewND<2, T4> & norm)
{
    vigra_precondition(1 == columnCount(mean) && rowCount(A) == rowCount(mean) &&
                       1 == columnCount(stdDev) && rowCount(A) == rowCount(stdDev) &&
                       1 == columnCount(norm) && rowCount(A) == rowCount(norm),
                       "rowStatistics(): Shape mismatch between input and output.");
    ArrayViewND<2, T2, StridedArrayTag> tm = transpose(mean);
    ArrayViewND<2, T3, StridedArrayTag> ts = transpose(stdDev);
    ArrayViewND<2, T4, StridedArrayTag> tn = transpose(norm);
    columnStatistics(transpose(A), tm, ts, tn);
}

namespace detail {

template <class T1, class U, class T2, class T3>
void updateCovarianceMatrix(ArrayViewND<2, T1> const & features,
                       U & count, ArrayViewND<2, T2> & mean, ArrayViewND<2, T3> & covariance)
{
    ArrayIndex n = std::max(rowCount(features), columnCount(features));
    vigra_precondition(std::min(rowCount(features), columnCount(features)) == 1,
          "updateCovarianceMatrix(): Features must be a row or column vector.");
    vigra_precondition(mean.shape() == features.shape(),
          "updateCovarianceMatrix(): Shape mismatch between feature vector and mean vector.");
    vigra_precondition(n == rowCount(covariance) && n == columnCount(covariance),
          "updateCovarianceMatrix(): Shape mismatch between feature vector and covariance matrix.");

    // West's algorithm for incremental covariance matrix computation
    Matrix<T2> t = features - mean;
    ++count;
    T2 f  = T2(1.0) / count,
       f1 = T2(1.0) - f;
    mean += f*t;

    if(rowCount(features) == 1) // update column covariance from current row
    {
        for(ArrayIndex k=0; k<n; ++k)
        {
            covariance(k, k) += f1*sq(t(0, k));
            for(ArrayIndex l=k+1; l<n; ++l)
            {
                covariance(k, l) += f1*t(0, k)*t(0, l);
                covariance(l, k) = covariance(k, l);
            }
        }
    }
    else // update row covariance from current column
    {
        for(ArrayIndex k=0; k<n; ++k)
        {
            covariance(k, k) += f1*sq(t(k, 0));
            for(ArrayIndex l=k+1; l<n; ++l)
            {
                covariance(k, l) += f1*t(k, 0)*t(l, 0);
                covariance(l, k) = covariance(k, l);
            }
        }
    }
}

} // namespace detail

    /** \brief Compute the covariance matrix between the columns of a matrix \a features.

        The result matrix \a covariance must by a square matrix with as many rows and
        columns as the number of columns in matrix \a features.

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2>
void covarianceMatrixOfColumns(ArrayViewND<2, T1> const & features,
                               ArrayViewND<2, T2> & covariance)
{
    ArrayIndex m = rowCount(features), n = columnCount(features);
    vigra_precondition(n == rowCount(covariance) && n == columnCount(covariance),
          "covarianceMatrixOfColumns(): Shape mismatch between feature matrix and covariance matrix.");
    ArrayIndex count = 0;
    Matrix<T2> means(1, n);
    covariance.init(NumericTraits<T2>::zero());
    for(ArrayIndex k=0; k<m; ++k)
        detail::updateCovarianceMatrix(rowVector(features, k), count, means, covariance);
    covariance /= T2(m - 1);
}

    /** \brief Compute the covariance matrix between the columns of a matrix \a features.

        The result is returned as a square temporary matrix with as many rows and
        columns as the number of columns in matrix \a features.

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T>
Matrix<T>
covarianceMatrixOfColumns(ArrayViewND<2, T> const & features)
{
    Matrix<T> res(columnCount(features), columnCount(features));
    covarianceMatrixOfColumns(features, res);
    return res;
}

    /** \brief Compute the covariance matrix between the rows of a matrix \a features.

        The result matrix \a covariance must by a square matrix with as many rows and
        columns as the number of rows in matrix \a features.

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T1, class T2>
void covarianceMatrixOfRows(ArrayViewND<2, T1> const & features,
                            ArrayViewND<2, T2> & covariance)
{
    ArrayIndex m = rowCount(features), n = columnCount(features);
    vigra_precondition(m == rowCount(covariance) && m == columnCount(covariance),
          "covarianceMatrixOfRows(): Shape mismatch between feature matrix and covariance matrix.");
    ArrayIndex count = 0;
    Matrix<T2> means(m, 1);
    covariance.init(NumericTraits<T2>::zero());
    for(ArrayIndex k=0; k<n; ++k)
        detail::updateCovarianceMatrix(columnVector(features, k), count, means, covariance);
    covariance /= T2(n - 1);
}

    /** \brief Compute the covariance matrix between the rows of a matrix \a features.

        The result is returned as a square temporary matrix with as many rows and
        columns as the number of rows in matrix \a features.

        <b>\#include</b> \<vigra2/matrix.hxx\><br>
        Namespace: vigra
    */
template <class T>
Matrix<T>
covarianceMatrixOfRows(ArrayViewND<2, T> const & features)
{
    Matrix<T> res(rowCount(features), rowCount(features));
    covarianceMatrixOfRows(features, res);
    return res;
}

enum DataPreparationGoals { ZeroMean = 1, UnitVariance = 2, UnitNorm = 4, UnitSum = 8 };

inline DataPreparationGoals operator|(DataPreparationGoals l, DataPreparationGoals r)
{
    return DataPreparationGoals(int(l) | int(r));
}

namespace detail {

template <class T>
void
prepareDataImpl(const ArrayViewND<2, T> & A,
               ArrayViewND<2, T> & res, ArrayViewND<2, T> & offset, ArrayViewND<2, T> & scaling,
               DataPreparationGoals goals)
{
    ArrayIndex m = rowCount(A);
    ArrayIndex n = columnCount(A);
    vigra_precondition(A.shape() == res.shape() &&
                       n == columnCount(offset) && 1 == rowCount(offset) &&
                       n == columnCount(scaling) && 1 == rowCount(scaling),
                       "prepareDataImpl(): Shape mismatch between input and output.");

    if(!goals)
    {
        res = A;
        offset.init(NumericTraits<T>::zero());
        scaling.init(NumericTraits<T>::one());
        return;
    }

    bool zeroMean = (goals & ZeroMean) != 0;
    bool unitVariance = (goals & UnitVariance) != 0;
    bool unitNorm = (goals & UnitNorm) != 0;
    bool unitSum = (goals & UnitSum) != 0;

    if(unitSum)
    {
        vigra_precondition(goals == UnitSum,
             "prepareData(): Unit sum is not compatible with any other data preparation goal.");

        transformMultiArray(srcMultiArrayRange(A), destMultiArrayRange(scaling), FindSum<T>());

        offset.init(NumericTraits<T>::zero());

        for(ArrayIndex k=0; k<n; ++k)
        {
            if(scaling(0, k) != NumericTraits<T>::zero())
            {
                scaling(0, k) = NumericTraits<T>::one() / scaling(0, k);
                columnVector(res, k) = columnVector(A, k) * scaling(0, k);
            }
            else
            {
                scaling(0, k) = NumericTraits<T>::one();
            }
        }

        return;
    }

    vigra_precondition(!(unitVariance && unitNorm),
        "prepareData(): Unit variance and unit norm cannot be achieved at the same time.");

    Matrix<T> mean(1, n), sumOfSquaredDifferences(1, n);
    detail::columnStatisticsImpl(A, mean, sumOfSquaredDifferences);

    for(ArrayIndex k=0; k<n; ++k)
    {
        T stdDev = std::sqrt(sumOfSquaredDifferences(0, k) / T(m-1));
        if(closeAtTolerance(stdDev / mean(0,k), NumericTraits<T>::zero()))
            stdDev = NumericTraits<T>::zero();
        if(zeroMean && stdDev > NumericTraits<T>::zero())
        {
            columnVector(res, k) = columnVector(A, k) - mean(0,k);
            offset(0, k) = mean(0, k);
            mean(0, k) = NumericTraits<T>::zero();
        }
        else
        {
            columnVector(res, k) = columnVector(A, k);
            offset(0, k) = NumericTraits<T>::zero();
        }

        T norm = mean(0,k) == NumericTraits<T>::zero()
                  ? std::sqrt(sumOfSquaredDifferences(0, k))
                  : std::sqrt(sumOfSquaredDifferences(0, k) + T(m) * sq(mean(0,k)));
        if(unitNorm && norm > NumericTraits<T>::zero())
        {
            columnVector(res, k) /= norm;
            scaling(0, k) = NumericTraits<T>::one() / norm;
        }
        else if(unitVariance && stdDev > NumericTraits<T>::zero())
        {
            columnVector(res, k) /= stdDev;
            scaling(0, k) = NumericTraits<T>::one() / stdDev;
        }
        else
        {
            scaling(0, k) = NumericTraits<T>::one();
        }
    }
}

} // namespace detail

    /** \brief Standardize the columns of a matrix according to given <tt>DataPreparationGoals</tt>.

    For every column of the matrix \a A, this function computes mean,
    standard deviation, and norm. It then applies a linear transformation to the values of
    the column according to these statistics and the given <tt>DataPreparationGoals</tt>.
    The result is returned in matrix \a res which must have the same size as \a A.
    Optionally, the transformation applied can also be returned in the matrices \a offset
    and \a scaling (see below for an example how these matrices can be used to standardize
    more data according to the same transformation).

    The following <tt>DataPreparationGoals</tt> are supported:

    <DL>
    <DT><tt>ZeroMean</tt><DD> Subtract the column mean form every column if the values in the column are not constant.
                              Do nothing in a constant column.
    <DT><tt>UnitSum</tt><DD> Scale the columns so that the their sum is one if the sum was initially non-zero.
                              Do nothing in a zero-sum column.
    <DT><tt>UnitVariance</tt><DD> Divide by the column standard deviation if the values in the column are not constant.
                              Do nothing in a constant column.
    <DT><tt>UnitNorm</tt><DD> Divide by the column norm if it is non-zero.
    <DT><tt>ZeroMean | UnitVariance</tt><DD> First subtract the mean and then divide by the standard deviation, unless the
                                             column is constant (in which case the column remains unchanged).
    <DT><tt>ZeroMean | UnitNorm</tt><DD> If the column is non-constant, subtract the mean. Then divide by the norm
                                         of the result if the norm is non-zero.
    </DL>

    <b> Declarations:</b>

    Standardize the matrix and return the parameters of the linear transformation.
    The matrices \a offset and \a scaling must be row vectors with as many columns as \a A.
    \code
    namespace vigra { namespace linalg {
        template <class T>
        void
        prepareColumns(ArrayViewND<2, T> const & A,
                       ArrayViewND<2, T> & res,
                       ArrayViewND<2, T> & offset,
                       ArrayViewND<2, T> & scaling,
                       DataPreparationGoals goals = ZeroMean | UnitVariance);
    } }
    \endcode

    Only standardize the matrix.
    \code
    namespace vigra { namespace linalg {
        template <class T>
        void
        prepareColumns(ArrayViewND<2, T> const & A,
                       ArrayViewND<2, T> & res,
                       DataPreparationGoals goals = ZeroMean | UnitVariance);
    } }
    \endcode

    <b> Usage:</b>

    <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
    <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
    Namespaces: vigra and vigra::linalg

    \code
    Matrix A(rows, columns);
    .. // fill A
    Matrix standardizedA(rows, columns), offset(1, columns), scaling(1, columns);

    prepareColumns(A, standardizedA, offset, scaling, ZeroMean | UnitNorm);

    // use offset and scaling to prepare additional data according to the same transformation
    Matrix newData(nrows, columns);

    Matrix standardizedNewData = (newData - repeatMatrix(offset, nrows, 1)) * pointWise(repeatMatrix(scaling, nrows, 1));

    \endcode
    */
doxygen_overloaded_function(template <...> void prepareColumns)

template <class T>
inline void
prepareColumns(ArrayViewND<2, T> const & A,
               ArrayViewND<2, T> & res, ArrayViewND<2, T> & offset, ArrayViewND<2, T> & scaling,
               DataPreparationGoals goals = ZeroMean | UnitVariance)
{
    detail::prepareDataImpl(A, res, offset, scaling, goals);
}

template <class T>
inline void
prepareColumns(ArrayViewND<2, T> const & A, ArrayViewND<2, T> & res,
               DataPreparationGoals goals = ZeroMean | UnitVariance)
{
    Matrix<T> offset(1, columnCount(A)), scaling(1, columnCount(A));
    detail::prepareDataImpl(A, res, offset, scaling, goals);
}

    /** \brief Standardize the rows of a matrix according to given <tt>DataPreparationGoals</tt>.

    This algorithm works in the same way as \ref prepareColumns() (see there for detailed
    documentation), but is applied to the rows of the matrix \a A instead. Accordingly, the
    matrices holding the parameters of the linear transformation must be column vectors
    with as many rows as \a A.

    <b> Declarations:</b>

    Standardize the matrix and return the parameters of the linear transformation.
    The matrices \a offset and \a scaling must be column vectors
    with as many rows as \a A.

    \code
    namespace vigra { namespace linalg {
        template <class T>
        void
        prepareRows(ArrayViewND<2, T> const & A,
                    ArrayViewND<2, T> & res,
                    ArrayViewND<2, T> & offset,
                    ArrayViewND<2, T> & scaling,
                    DataPreparationGoals goals = ZeroMean | UnitVariance)´;
    } }
    \endcode

    Only standardize the matrix.
    \code
    namespace vigra { namespace linalg {
        template <class T>
        void
        prepareRows(ArrayViewND<2, T> const & A,
                    ArrayViewND<2, T> & res,
                    DataPreparationGoals goals = ZeroMean | UnitVariance);
    } }
    \endcode

    <b> Usage:</b>

    <b>\#include</b> \<vigra2/matrix.hxx\> or<br>
    <b>\#include</b> \<vigra2/linear_algebra.hxx\><br>
    Namespaces: vigra and vigra::linalg

    \code
    Matrix A(rows, columns);
    .. // fill A
    Matrix standardizedA(rows, columns), offset(rows, 1), scaling(rows, 1);

    prepareRows(A, standardizedA, offset, scaling, ZeroMean | UnitNorm);

    // use offset and scaling to prepare additional data according to the same transformation
    Matrix newData(rows, ncolumns);

    Matrix standardizedNewData = (newData - repeatMatrix(offset, 1, ncolumns)) * pointWise(repeatMatrix(scaling, 1, ncolumns));

    \endcode
*/
doxygen_overloaded_function(template <...> void prepareRows)

template <class T>
inline void
prepareRows(ArrayViewND<2, T> const & A,
            ArrayViewND<2, T> & res, ArrayViewND<2, T> & offset, ArrayViewND<2, T> & scaling,
            DataPreparationGoals goals = ZeroMean | UnitVariance)
{
    ArrayViewND<2, T, StridedArrayTag> tr = transpose(res), to = transpose(offset), ts = transpose(scaling);
    detail::prepareDataImpl(transpose(A), tr, to, ts, goals);
}

template <class T>
inline void
prepareRows(ArrayViewND<2, T> const & A, ArrayViewND<2, T> & res,
            DataPreparationGoals goals = ZeroMean | UnitVariance)
{
    ArrayViewND<2, T, StridedArrayTag> tr = transpose(res);
    Matrix<T> offset(1, rowCount(A)), scaling(1, rowCount(A));
    detail::prepareDataImpl(transpose(A), tr, offset, scaling, goals);
}

//@}

} // namespace linalg

using linalg::columnStatistics;
using linalg::prepareColumns;
using linalg::rowStatistics;
using linalg::prepareRows;
using linalg::ZeroMean;
using linalg::UnitVariance;
using linalg::UnitNorm;
using linalg::UnitSum;

}  // namespace vigra



#endif // VIGRA2_MATH_MATRIX_HXX
