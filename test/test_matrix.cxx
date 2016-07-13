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

#include <typeinfo>
#include <iostream>
#include <string>
#include <random>
#include <vigra2/unittest.hxx>
#include <vigra2/math/matrix.hxx>

using namespace vigra;


struct MatrixTest
{
    typedef Matrix<double> Matrix;
    typedef Matrix::difference_type Shape;

    unsigned int size, iterations;
    std::mt19937 random_;
    std::uniform_real_distribution<> random_double_;

    MatrixTest()
    : size(50)
    , iterations(5)
    , random_(23098349)
    , random_double_(-1.0, 1.0)
    {}

    // void testOStreamShifting()
    // {
        // Matrix a = random_matrix (size, size);
        // std::ostringstream out;
        // out << a;
        // out << "Testing.." << a << 42 << std::endl;
    // }

    // double random_double ()
    // {
        // double ret = 2.0 * random_.uniform53() - 1.0;
        // return ret;
    // }

     Matrix random_matrix(unsigned int rows, unsigned int cols)
     {
         Matrix ret (rows, cols);
         for (unsigned int i = 0; i < rows; ++i)
             for (unsigned int j = 0; j < cols; ++j)
                 ret (i, j) = random_double_(random_);
         return ret;
     }

     Matrix random_symmetric_matrix(unsigned int rows)
     {
         Matrix ret (rows, rows);
         for (unsigned int i = 0; i < rows; ++i)
             for (unsigned int j = i; j < rows; ++j)
                 ret (j, i) = ret (i, j) = random_double_(random_);
         return ret;
     }

    void testMatrix()
    {
        double data[] = {1.0, 5.0,
                         3.0, 2.0,
                         4.0, 7.0};
        double tref[] = {1.0, 3.0, 4.0,
                         5.0, 2.0, 7.0};
        double tref2[] = {1.0, 3.0,
                          5.0, 2.0};
        double idref[] = {1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0};
        std::string sref("1.0000 5.0000 \n3.0000 2.0000 \n4.0000 7.0000 \n");
        unsigned int r = 3, c = 2;

        Matrix a({ r, c }, data), zero(r, c);
        shouldEqual(rowCount(a), r);
        shouldEqual(columnCount(a), c);
        shouldEqual(elementCount(a), r*c);
        shouldEqual(squaredNorm(a), 104.0);
        shouldEqual(norm(a), std::sqrt(104.0));

        using namespace linalg;

        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(zero(i,j), 0.0);

        Matrix one = zero + Matrix({ r,c }, 1.0);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(one(i,j), 1.0);

        std::stringstream s;
        s << std::setprecision(4) << a;
        shouldEqual(s.str(), sref);

        for(unsigned int i=0, k=0; i<r; ++i)
        {
            Matrix::view_type ar = rowVector(a, i);
            shouldEqual(rowCount(ar), 1);
            shouldEqual(columnCount(ar), c);
            for(unsigned int j=0; j<c; ++j, ++k)
            {
                shouldEqual(a(i,j), data[k]);
                shouldEqual(ar(0, j), data[k]);
            }
        }

        Matrix aa({ r, c }, tref, ColumnMajor);
        shouldEqual(rowCount(aa), r);
        shouldEqual(columnCount(aa), c);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(aa(i,j), a(i,j));

        Matrix b = a;
        shouldEqual(rowCount(b), r);
        shouldEqual(columnCount(b), c);
        shouldEqualSequence(a.begin(), a.end(), b.begin());

        b = 0.0;
        should(b == zero);

        Matrix::iterator ib = b.begin();
        b = a;
        should(ib == b.begin());
        shouldEqualSequence(a.begin(), a.end(), b.begin());

        b = 4.0 + a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 4.0+data[k]);
        b = a + 3.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k]+3.0);
        b += 4.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 7.0+data[k]);
        b += a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 7.0+2.0*data[k]);


        b = 4.0 - a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 4.0-data[k]);
        b = a - 3.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k]-3.0);
        b -= 4.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k]-7.0);
        b -= a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), -7.0);

        b = 4.0 * a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 4.0*data[k]);
        b = a * 3.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k]*3.0);
        b *= 4.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k]*12.0);
        b *= a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k]*data[k]*12.0);

        b = 4.0 / a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 4.0/data[k]);
        b = a / 3.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k] / 3.0);
        b /= 4.0;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k] / 12.0);
        b /= a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqualTolerance(b(i,j), 1.0 / 12.0, 1e-12);

        b = a + a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), 2.0 * data[k]);

        b = a - a;
        for(unsigned int i=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j)
                shouldEqual(b(i,j), 0.0);

        b = -a;
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), -data[k]);

        b = a * pointWise(a);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k] * data[k]);

        b = a / pointWise(a);
        for(unsigned int i=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j)
                shouldEqual(b(i,j), 1.0);

        b = pow(a, 2);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), data[k] * data[k]);

        b = sqrt(a);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), sqrt(data[k]));

        b = sq(a);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), sq(data[k]));

        b = sign(a);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(b(i,j), sign(data[k]));

        Matrix at = transpose(a);
        shouldEqual(rowCount(at), c);
        shouldEqual(columnCount(at), r);
        for(unsigned int i=0, k=0; i<c; ++i)
        {
            Matrix::view_type ac = columnVector(a, i);
            shouldEqual(rowCount(ac), r);
            shouldEqual(columnCount(ac), 1);
            for(unsigned int j=0; j<r; ++j, ++k)
            {
                shouldEqual(at(i,j), tref[k]);
                shouldEqual(ac(j,0), tref[k]);
            }
            shouldEqual(ac, subVector(ac, 0, r));
            shouldEqual(a.subarray(Shape(1, i), Shape(r-1, i+1)), subVector(ac, 1, r-1));
        }

        double sn = squaredNorm(columnVector(a, 0));
        shouldEqual(sn, 26.0);
        shouldEqual(sn, dot(columnVector(a, 0), columnVector(a, 0)));
        shouldEqual(sn, dot(rowVector(at, 0), columnVector(a, 0)));
        shouldEqual(sn, dot(columnVector(a, 0), rowVector(at, 0)));
        shouldEqual(sn, dot(rowVector(at, 0), rowVector(at, 0)));
        shouldEqual(0.0, dot(a.subarray(Shape(0,0), Shape(1,0)), a.subarray(Shape(0,0), Shape(0,1))));
        shouldEqual(0.0, dot(a.subarray(Shape(0,0), Shape(0,1)), a.subarray(Shape(0,0), Shape(0,1))));
        shouldEqual(0.0, dot(a.subarray(Shape(0,0), Shape(1,0)), a.subarray(Shape(0,0), Shape(1,0))));
        shouldEqual(0.0, dot(a.subarray(Shape(0,0), Shape(0,1)), a.subarray(Shape(0,0), Shape(1,0))));

        Matrix a2({ c, c }, data);
        a2 = a2.transpose();
        for(unsigned int i=0, k=0; i<c; ++i)
            for(unsigned int j=0; j<c; ++j, ++k)
                shouldEqual(a2(i,j), tref2[k]);

        shouldEqual(trace(a2), 3.0);

        Matrix id = identityMatrix<double>(r);
        shouldEqual(rowCount(id), r);
        shouldEqual(columnCount(id), r);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<r; ++j, ++k)
                shouldEqual(id(i,j), idref[k]);

        shouldEqual(trace(id), 3.0);

        Matrix d = diagonalMatrix(Matrix({ r, 1 }, data));
        shouldEqual(rowCount(d), r);
        shouldEqual(columnCount(d), r);
        for(unsigned int i=0, k=0; i<r; ++i)
            for(unsigned int j=0; j<r; ++j, ++k)
                shouldEqual(d(i,j), idref[k]*data[i]);

        Matrix e({ r*c, 1 }, data);
        shouldEqual(dot(transpose(e), e), squaredNorm(e));

        double dc1[] = {1.0, 1.0, 1.0},
               dc2[] = {1.2, 2.4, 3.6};
        Matrix c1({ 3,1 }, dc1), c2({ 3, 1 }, dc2);
        Matrix cr = cross(c1, c2);
        shouldEqualTolerance(cr(0,0), 1.2, 1e-12);
        shouldEqualTolerance(cr(1,0), -2.4, 1e-12);
        shouldEqualTolerance(cr(2,0), 1.2, 1e-12);

        Matrix f({ 1, r*c - 1 }, tref);
        Matrix g = outer(e, f);
        shouldEqual(rowCount(g), rowCount(e));
        shouldEqual(columnCount(g), columnCount(f));
        for(int i=0; i<rowCount(g); ++i)
            for(int j=0; j<columnCount(g); ++j)
                shouldEqual(g(i,j), data[i]*tref[j]);

        Matrix g1 = outer(e);
        shouldEqual(rowCount(g1), rowCount(e));
        shouldEqual(columnCount(g1), rowCount(e));
        for(int i=0; i<rowCount(g1); ++i)
            for(int j=0; j<columnCount(g1); ++j)
                shouldEqual(g1(i,j), data[i]*data[j]);

        Matrix g2 = outer(TinyArray<double, 6>(data));
        shouldEqual(rowCount(g2), 6);
        shouldEqual(columnCount(g2), 6);
        for(int i=0; i<rowCount(g2); ++i)
            for(int j=0; j<columnCount(g2); ++j)
                shouldEqual(g2(i,j), data[i]*data[j]);

        Matrix h = transpose(a) * a;
        shouldEqual(rowCount(h), c);
        shouldEqual(columnCount(h), c);
        for(int i=0; i<(int)c; ++i)
            for(int j=0; j<(int)c; ++j)
                shouldEqual(h(i,j), dot(rowVector(at, i), columnVector(a, j)));

        should(isSymmetric(random_symmetric_matrix(10)));
        should(!isSymmetric(random_matrix(10, 10)));

        Matrix tm({ 2, 2 }, tref2);
        TinyArray<double, 2> tv(1.0, 2.0), tvrref(7.0, 9.0), tvlref(11.0, 7.0);
        shouldEqual(tm * tv, tvrref);
        shouldEqual(tv * tm, tvlref);

        Matrix rep = repeatMatrix(a, 2, 4);
        shouldEqual(rowCount(rep), 2*r);
        shouldEqual(columnCount(rep), 4*c);

        for(unsigned int l=0; l<4; ++l)
            for(unsigned int k=0; k<2; ++k)
                for(unsigned int j=0; j<c; ++j)
                    for(unsigned int i=0; i<r; ++i)
                        shouldEqual(rep(k*r+i, l*c+j), a(i,j));

        double columnSum[] = {8.0, 14.0};
        double rowSum[] = {6.0, 5.0, 11.0};
        auto as0 = a.sum(tags::axis = 0);
        auto as1 = a.sum(tags::axis = 1);
        shouldEqualSequence(columnSum, columnSum+2, as0.begin());
        shouldEqualSequence(rowSum, rowSum+3, as1.begin());

        double columnMean[] = {8/3.0, 14/3.0};
        double rowMean[] = {3.0, 2.5, 5.5};
        auto am0 = a.mean(tags::axis = 0);
        auto am1 = a.mean(tags::axis = 1);
        shouldEqualSequence(columnMean, columnMean+2, am0.begin());
        shouldEqualSequence(rowMean, rowMean+3, am1.begin());
    }

     void testArgMinMax()
     {
         double data[] = {1.0, 5.0,
                          3.0, 2.0,
                         -2.0, 4.0};
         unsigned int r = 3, c = 2;
         Matrix minmax({ r, c }, data);

         shouldEqual(argMin(minmax), 4);
         shouldEqual(argMax(minmax), 1);
         shouldEqual(argMinIf(minmax, [](double v) { return v > 0.0; }), 0);
         shouldEqual(argMinIf(minmax, [](double v) { return v > 5.0; }), -1);
         shouldEqual(argMaxIf(minmax, [](double v) { return v < 5.0; }), 5);
         shouldEqual(argMaxIf(minmax, [](double v) { return v < -2.0; }), -1);
     }

    // void testColumnAndRowStatistics()
    // {
        // double epsilon = 1e-11;

        // Matrix rowMean(size, 1), columnMean(1, size);
        // Matrix rowStdDev(size, 1), columnStdDev(1, size);
        // Matrix rowNorm(size, 1), columnNorm(1, size);
        // Matrix rowCovariance(size, size), columnCovariance(size, size);

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);

            // rowStatistics(a, rowMean, rowStdDev, rowNorm);
            // columnStatistics(a, columnMean, columnStdDev, columnNorm);

            // for(unsigned int k=0; k<size; ++k)
            // {
                // double rm = 0.0, cm = 0.0, rn = 0.0, cn = 0.0, rs = 0.0, cs = 0.0;
                // for(unsigned int l=0; l<size; ++l)
                // {
                    // rm += a(k, l);
                    // cm += a(l, k);
                    // rn += sq(a(k, l));
                    // cn += sq(a(l, k));
                // }
                // rm /= size;
                // cm /= size;
                // rn = std::sqrt(rn);
                // cn = std::sqrt(cn);

                // shouldEqualTolerance(rm, rowMean(k,0), epsilon);
                // shouldEqualTolerance(cm, columnMean(0,k), epsilon);
                // shouldEqualTolerance(rn, rowNorm(k,0), epsilon);
                // shouldEqualTolerance(cn, columnNorm(0,k), epsilon);

                // for(unsigned int l=0; l<size; ++l)
                // {
                    // rs += sq(a(k, l) - rm);
                    // cs += sq(a(l, k) - cm);
                // }
                // rs = std::sqrt(rs / (size-1));
                // cs = std::sqrt(cs / (size-1));

                // shouldEqualTolerance(rs, rowStdDev(k,0), epsilon);
                // shouldEqualTolerance(cs, columnStdDev(0,k), epsilon);
            // }

            // covarianceMatrixOfRows(a, rowCovariance);
            // covarianceMatrixOfColumns(a, columnCovariance);
            // Matrix rowCovarianceRef(size, size), columnCovarianceRef(size, size);
            // for(unsigned int k=0; k<size; ++k)
            // {
                // for(unsigned int l=0; l<size; ++l)
                // {
                    // for(unsigned int m=0; m<size; ++m)
                    // {
                        // rowCovarianceRef(l, m) += (a(l, k) - rowMean(l, 0)) * (a(m, k) - rowMean(m, 0));
                        // columnCovarianceRef(l, m) += (a(k, l) - columnMean(0, l)) * (a(k, m) - columnMean(0, m));
                    // }
                // }
            // }
            // rowCovarianceRef /= (size-1);
            // columnCovarianceRef /= (size-1);

            // shouldEqualSequenceTolerance(rowCovariance.data(), rowCovariance.data()+size*size, rowCovarianceRef.data(), epsilon);
            // shouldEqualSequenceTolerance(columnCovariance.data(), columnCovariance.data()+size*size, columnCovarianceRef.data(), epsilon);
        // }
    // }

    // void testColumnAndRowPreparation()
    // {
        // using ZeroMean;
        // using UnitVariance;
        // using UnitNorm;
        // using UnitSum;

        // double epsilon = 1e-11;

        // Matrix rowMean(size, 1), columnMean(1, size);
        // Matrix rowStdDev(size, 1), columnStdDev(1, size);
        // Matrix rowNorm(size, 1), columnNorm(1, size);

        // Matrix rowPrepared(size, size), columnPrepared(size, size);
        // Matrix rowMeanPrepared(size, 1), columnMeanPrepared(1, size);
        // Matrix rowStdDevPrepared(size, 1), columnStdDevPrepared(1, size);
        // Matrix rowNormPrepared(size, 1), columnNormPrepared(1, size);
        // Matrix rowOffset(size, 1), columnOffset(1, size);
        // Matrix rowScaling(size, 1), columnScaling(1, size);

        // Matrix zeroRowRef(size,1), zeroColRef(1, size);
        // Matrix oneRowRef(size,1), oneColRef(1, size);
        // oneRowRef.init(1.0);
        // oneColRef.init(1.0);

        // {
            // Matrix a = random_matrix (size, size);

            // columnStatistics(a, columnMean, columnStdDev, columnNorm);

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, UnitSum);
            // shouldEqualSequence(zeroColRef.data(), zeroColRef.data()+size, columnOffset.data());
            // columnScaling *= columnMean;
            // columnScaling *= size;
            // shouldEqualSequenceTolerance(oneColRef.data(), oneColRef.data()+size, columnScaling.data(), epsilon);
            // columnStatistics(columnPrepared, columnMeanPrepared, columnStdDevPrepared, columnNormPrepared);
            // columnMeanPrepared *= size;
            // shouldEqualSequenceTolerance(oneColRef.data(), oneColRef.data()+size, columnMeanPrepared.data(), epsilon);

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, ZeroMean);
            // columnStatistics(columnPrepared, columnMeanPrepared, columnStdDevPrepared, columnNormPrepared);
            // shouldEqualSequenceTolerance(zeroColRef.data(), zeroColRef.data()+size, columnMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(columnStdDev.data(), columnStdDev.data()+size, columnStdDevPrepared.data(), epsilon);

            // Matrix ap = columnPrepared / pointWise(repeatMatrix(columnScaling, size, 1)) + repeatMatrix(columnOffset, size, 1);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, UnitNorm);
            // columnStatistics(columnPrepared, columnMeanPrepared, columnStdDevPrepared, columnNormPrepared);
            // shouldEqualSequenceTolerance(oneColRef.data(), oneColRef.data()+size, columnNormPrepared.data(), epsilon);

            // ap = columnPrepared / pointWise(repeatMatrix(columnScaling, size, 1)) + repeatMatrix(columnOffset, size, 1);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, UnitVariance);
            // columnStatistics(columnPrepared, columnMeanPrepared, columnStdDevPrepared, columnNormPrepared);
            // columnMeanPrepared /= columnScaling;
            // shouldEqualSequenceTolerance(columnMean.data(), columnMean.data()+size, columnMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(oneColRef.data(), oneColRef.data()+size, columnStdDevPrepared.data(), epsilon);

            // ap = columnPrepared / pointWise(repeatMatrix(columnScaling, size, 1)) + repeatMatrix(columnOffset, size, 1);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, ZeroMean | UnitVariance);
            // columnStatistics(columnPrepared, columnMeanPrepared, columnStdDevPrepared, columnNormPrepared);
            // shouldEqualSequenceTolerance(zeroColRef.data(), zeroColRef.data()+size, columnMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(oneColRef.data(), oneColRef.data()+size, columnStdDevPrepared.data(), epsilon);

            // ap = columnPrepared / pointWise(repeatMatrix(columnScaling, size, 1)) + repeatMatrix(columnOffset, size, 1);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, ZeroMean | UnitNorm);
            // columnStatistics(columnPrepared, columnMeanPrepared, columnStdDevPrepared, columnNormPrepared);
            // shouldEqualSequenceTolerance(zeroColRef.data(), zeroColRef.data()+size, columnMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(oneColRef.data(), oneColRef.data()+size, columnNormPrepared.data(), epsilon);

            // ap = columnPrepared / pointWise(repeatMatrix(columnScaling, size, 1)) + repeatMatrix(columnOffset, size, 1);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // rowStatistics(a, rowMean, rowStdDev, rowNorm);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, UnitSum);
            // shouldEqualSequence(zeroRowRef.data(), zeroRowRef.data()+size, rowOffset.data());
            // rowScaling *= rowMean;
            // rowScaling *= size;
            // shouldEqualSequenceTolerance(oneRowRef.data(), oneRowRef.data()+size, rowScaling.data(), epsilon);
            // rowStatistics(rowPrepared, rowMeanPrepared, rowStdDevPrepared, rowNormPrepared);
            // rowMeanPrepared *= size;
            // shouldEqualSequenceTolerance(oneRowRef.data(), oneRowRef.data()+size, rowMeanPrepared.data(), epsilon);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, ZeroMean);
            // rowStatistics(rowPrepared, rowMeanPrepared, rowStdDevPrepared, rowNormPrepared);
            // shouldEqualSequenceTolerance(zeroRowRef.data(), zeroRowRef.data()+size, rowMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(rowStdDev.data(), rowStdDev.data()+size, rowStdDevPrepared.data(), epsilon);

            // ap = rowPrepared / pointWise(repeatMatrix(rowScaling, 1, size)) + repeatMatrix(rowOffset, 1, size);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, UnitNorm);
            // rowStatistics(rowPrepared, rowMeanPrepared, rowStdDevPrepared, rowNormPrepared);
            // shouldEqualSequenceTolerance(oneRowRef.data(), oneRowRef.data()+size, rowNormPrepared.data(), epsilon);

            // ap = rowPrepared / pointWise(repeatMatrix(rowScaling, 1, size)) + repeatMatrix(rowOffset, 1, size);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, UnitVariance);
            // rowStatistics(rowPrepared, rowMeanPrepared, rowStdDevPrepared, rowNormPrepared);
            // rowMeanPrepared /= rowScaling;
            // shouldEqualSequenceTolerance(rowMean.data(), rowMean.data()+size, rowMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(oneRowRef.data(), oneRowRef.data()+size, rowStdDevPrepared.data(), epsilon);

            // ap = rowPrepared / pointWise(repeatMatrix(rowScaling, 1, size)) + repeatMatrix(rowOffset, 1, size);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, ZeroMean | UnitVariance);
            // rowStatistics(rowPrepared, rowMeanPrepared, rowStdDevPrepared, rowNormPrepared);
            // shouldEqualSequenceTolerance(zeroRowRef.data(), zeroRowRef.data()+size, rowMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(oneRowRef.data(), oneRowRef.data()+size, rowStdDevPrepared.data(), epsilon);

            // ap = rowPrepared / pointWise(repeatMatrix(rowScaling, 1, size)) + repeatMatrix(rowOffset, 1, size);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, ZeroMean | UnitNorm);
            // rowStatistics(rowPrepared, rowMeanPrepared, rowStdDevPrepared, rowNormPrepared);
            // shouldEqualSequenceTolerance(zeroRowRef.data(), zeroRowRef.data()+size, rowMeanPrepared.data(), epsilon);
            // shouldEqualSequenceTolerance(oneRowRef.data(), oneRowRef.data()+size, rowNormPrepared.data(), epsilon);

            // ap = rowPrepared / pointWise(repeatMatrix(rowScaling, 1, size)) + repeatMatrix(rowOffset, 1, size);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);
        // }

        // {
            // Matrix a(size, size, 2.0), aref(size, size, 1.0/std::sqrt((double)size));

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, ZeroMean | UnitVariance);
            // shouldEqualSequence(a.data(), a.data()+size*size, columnPrepared.data());

            // prepareColumns(a, columnPrepared, columnOffset, columnScaling, ZeroMean | UnitNorm);
            // shouldEqualSequenceTolerance(aref.data(), aref.data()+size*size, columnPrepared.data(), epsilon);
            // Matrix ap = columnPrepared / pointWise(repeatMatrix(columnScaling, size, 1)) + repeatMatrix(columnOffset, size, 1);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, ZeroMean | UnitVariance);
            // shouldEqualSequence(a.data(), a.data()+size*size, rowPrepared.data());

            // prepareRows(a, rowPrepared, rowOffset, rowScaling, ZeroMean | UnitNorm);
            // shouldEqualSequenceTolerance(aref.data(), aref.data()+size*size, rowPrepared.data(), epsilon);
            // ap = rowPrepared / pointWise(repeatMatrix(rowScaling, 1, size)) + repeatMatrix(rowOffset, 1, size);
            // shouldEqualSequenceTolerance(a.data(), a.data()+size*size, ap.data(), epsilon);
        // }
    // }

    // void testCholesky()
    // {
        // double epsilon = 1e-11;
        // Matrix idref = identityMatrix<double>(size);

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);
            // a = transpose(a) * a; // make a symmetric positive definite matrix
            // Matrix l(size, size);
            // choleskyDecomposition (a, l);
            // Matrix ch = l * transpose(l);
            // shouldEqualSequenceTolerance(ch.data(), ch.data()+size*size, a.data(), epsilon);
        // }
    // }

    // void testQR()
    // {
        // double epsilon = 1e-11;
        // Matrix idref = identityMatrix<double>(size);

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);
            // Matrix r(size, size);
            // Matrix q(size, size);
            // qrDecomposition (a, q, r);
            // Matrix id = transpose(q) * q;
            // shouldEqualSequenceTolerance(id.data(), id.data()+size*size, idref.data(), epsilon);
            // Matrix qr = q * r;
            // shouldEqualSequenceTolerance(qr.data(), qr.data()+size*size, a.data(), epsilon);
        // }
    // }

    // void testLinearSolve()
    // {
        // double epsilon = 1e-11;
        // int size = 50;

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);
            // Matrix b = random_matrix (size, 1);
            // Matrix x(size, 1);

            // should(linearSolve (a, b, x, "QR"));
            // Matrix ax = a * x;
            // shouldEqualSequenceTolerance(ax.data(), ax.data()+size, b.data(), epsilon);

            // should(linearSolve(a, b, x, "SVD"));
            // ax = a * x;
            // shouldEqualSequenceTolerance(ax.data(), ax.data()+size, b.data(), epsilon);

            // should(linearSolve(a, b, x, "NE"));
            // ax = a * x;
            // shouldEqualSequenceTolerance(ax.data(), ax.data()+size, b.data(), epsilon);

            // Matrix c = transpose(a) * a; // make a symmetric positive definite matrix
            // Matrix d = transpose(a) * b;
            // should(linearSolve (c, d, x, "Cholesky"));
            // ax = c * x;
            // shouldEqualSequenceTolerance(ax.data(), ax.data()+size, d.data(), epsilon);
        // }

        // size = 4;
        // Matrix a = random_matrix (size, size);
        // Matrix b = random_matrix (size, 1);
        // Matrix x(size, 1);

        // TinyArray<double, 4> vb(b.data()), vx;
        // should(linearSolve (a, b, x));
        // should(linearSolve (a, vb, vx));
        // shouldEqualSequenceTolerance(x.data(), x.data()+size, vx.data(), epsilon);
    // }

    // void testUnderdetermined()
    // {
        // // test singular matrix
        // Matrix a = identityMatrix<Matrix::value_type> (size);
        // a(0,0) = 0;
        // Matrix b = random_matrix (size, 1);
        // Matrix x(size, 1);
        // should(!linearSolve (a, b, x, "Cholesky"));
        // should(!linearSolve (a, b, x, "QR"));
        // should(!linearSolve (a, b, x, "SVD"));

        // {
            // // square, rank-deficient system (compute minimum norm solution)
            // double mdata[] = {1.0,  3.0,  7.0,
                             // -1.0,  4.0,  4.0,
                              // 1.0, 10.0, 18.0};
            // double rhsdata[] = { 5.0, 2.0, 12.0};
            // double refdata[] = { 0.3850, -0.1103, 0.7066 };

            // Matrix m(3,3,mdata), rhs(3,1,rhsdata), xx(3,1);

            // shouldEqual(linearSolveQR(m, rhs, xx), 2u);
            // shouldEqualSequenceTolerance(refdata, refdata+3, xx.data(), 1e-3);
        // }
        // {
            // // underdetermined, full-rank system (compute minimum norm solution)
            // double mdata[] = {2.0, -3.0, 1.0, -6.0,
                              // 4.0,  1.0, 2.0,  9.0,
                              // 3.0,  1.0, 1.0,  8.0};
            // double rhsdata[] = { -7.0, -7.0, -8.0};
            // double refdata[] = { -3.26666666666667, 3.6, 5.13333333333333, -0.86666666666667 };

            // Matrix m(3,4,mdata), rhs(3,1,rhsdata), xx(4,1);

            // shouldEqual(linearSolveQR(m, rhs, xx), 3u);
            // shouldEqualSequenceTolerance(refdata, refdata+4, xx.data(), 1e-12);
        // }
        // {
            // // underdetermined, rank-deficient, consistent system (compute minimum norm solution)
            // double mdata[] = {1.0,  3.0, 3.0, 2.0,
                              // 2.0,  6.0, 9.0, 5.0,
                             // -1.0, -3.0, 3.0, 0.0};
            // double rhsdata[] = { 1.0, 5.0, 5.0};
            // double refdata[] = { -0.211009, -0.633027, 0.963303, 0.110092 };

            // Matrix m(3,4,mdata), rhs(3,1,rhsdata), xx(4,1);

            // shouldEqual(linearSolveQR(m, rhs, xx), 2u);
            // shouldEqualSequenceTolerance(refdata, refdata+4, xx.data(), 1e-5);
        // }
        // {
            // // underdetermined, rank-deficient, inconsistent system (compute minimum norm least squares solution)
            // double mdata[] = {2.0, 1.0,  7.0, -7.0,
                             // -3.0, 4.0, -5.0, -6.0,
                              // 1.0, 1.0,  4.0, -5.0};
            // double rhsdata[] = { 2.0, 3.0, 2.0};
            // double refdata[] = { -0.0627, 0.1561, -0.0321, -0.3427 };

            // Matrix m(3,4,mdata), rhs(3,1,rhsdata), xx(4,1);

            // shouldEqual(linearSolveQR(m, rhs, xx), 2u);
            // shouldEqualSequenceTolerance(refdata, refdata+4, xx.data(), 1e-3);
        // }
    // }

    // void testOverdetermined()
    // {
        // double epsilon = 1e-11;

        // unsigned int n = 5;
        // unsigned int size = 1000;
        // double noiseStdDev = 0.1;

        // Matrix A(size, n), xs(n,1), xq(n,1), xn(n,1), r(size, 1);

        // for(unsigned int iter=0; iter<iterations; ++iter)
        // {
            // // set up a linear regression problem for a polynomial of degree n
            // Matrix weights = random_matrix (n, 1);
            // Matrix v = random_matrix (size, 1);

            // // init rhs with Gaussian noise with zero mean and noiseStdDev
            // Matrix rhs = 0.5*noiseStdDev*random_matrix (size, 1);
            // for(unsigned int k=1; k<12; ++k)
                // rhs += 0.5*noiseStdDev*random_matrix (size, 1);

            // for(unsigned int k=0; k<size; ++k)
            // {
                // for(unsigned int l=0; l<n; ++l)
                // {
                    // A(k,l) = std::pow(v(k,0), double(l));
                    // rhs(k,0) += weights(l,0)*A(k,l);
                // }
            // }

            // shouldEqual(linearSolve(A, rhs, xs, "SVD"), true);

            // // check that solution is indeed a minimum by
            // // testing for zero derivative of the objective
            // Matrix derivative = abs(transpose(A)*(A*xs - rhs));
            // int absIndex = argMax(derivative);
            // shouldEqualTolerance(derivative(absIndex,0), 0.0, epsilon);

            // shouldEqual(linearSolveQR(A, rhs, xq), n);
            // shouldEqualSequenceTolerance(xs.data(), xs.data()+n, xq.data(), epsilon);

            // shouldEqual(linearSolve(A, rhs, xn, "ne"), true);
            // shouldEqualSequenceTolerance(xs.data(), xs.data()+n, xn.data(), epsilon);
        // }
    // }

    // void testIncrementalLinearSolve()
    // {
        // double epsilon = 1e-11;
        // int size = 50;

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);
            // Matrix b = random_matrix (size, 1);
            // Matrix x(size, 1);

            // should(linearSolve(a, b, x, "QR"));

            // {
                // Matrix r(a), qtb(b), px(size,1), xx(size,1);
                // std::vector<unsigned int> permutation(size);

                // for(int k=0; k<size; ++k)
                // {
                    // // use Givens steps for a change (Householder steps like
                    // //    should(linalg::detail::qrColumnHouseholderStep(k, r, qtb));
                    // // work as well, but are already extensively tested within the QR algorithm)
                    // should(linalg::detail::qrGivensStepImpl(k, r, qtb));
                    // permutation[k] = k;
                // }

                // for(int k=0; k<size; ++k)
                // {
                    // int i = random_.uniformInt(size), j = random_.uniformInt(size);
                    // if(i==j) continue;

                    // linalg::detail::upperTriangularCyclicShiftColumns(i, j, r, qtb, permutation);
                // }
                // should(linalg::linearSolveUpperTriangular(r, qtb, px));
                // linalg::detail::inverseRowPermutation(px, xx, permutation);

                // shouldEqualSequenceTolerance(x.data(), x.data()+size, xx.data(), epsilon);
            // }

            // {
                // Matrix r(a), qtb(b), px(size,1), xx(size,1);
                // std::vector<unsigned int> permutation(size);

                // for(int k=0; k<size; ++k)
                // {
                    // // use Givens steps for a change (Householder steps like
                    // //    should(linalg::detail::qrColumnHouseholderStep(k, r, qtb));
                    // // work as well, but are already extensively tested within the QR algorithm)
                    // should(linalg::detail::qrGivensStepImpl(k, r, qtb));
                    // permutation[k] = k;
                // }

                // for(int k=0; k<size; ++k)
                // {
                    // int i = random_.uniformInt(size), j = random_.uniformInt(size);
                    // linalg::detail::upperTriangularSwapColumns(i, j, r, qtb, permutation);
                // }
                // should(linalg::linearSolveUpperTriangular(r, qtb, px));
                // linalg::detail::inverseRowPermutation(px, xx, permutation);

                // shouldEqualSequenceTolerance(x.data(), x.data()+size, xx.data(), epsilon);
            // }
        // }
    // }

    // void testInverse()
    // {
        // double epsilon = 1e-11;
        // Matrix idref = identityMatrix<double>(size);

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);
            // Matrix id = a * inverse(a);
            // shouldEqualSequenceTolerance(id.data(), id.data()+size*size, idref.data(), epsilon);
            // id = inverse(a) * a;
            // shouldEqualSequenceTolerance(id.data(), id.data()+size*size, idref.data(), epsilon);
            // id = inverse(idref * a) * a; // test inverse(const TemporaryMatrix<T> &v)
            // shouldEqualSequenceTolerance(id.data(), id.data()+size*size, idref.data(), epsilon);
        // }

        // double data[] = { 1.0, 0.0, 0.0, 0.0 };
        // Matrix singular(2, 2, data);
        // try {
            // inverse(singular);
            // failTest("inverse(singular) didn't throw an exception.");
        // }
        // catch(PreconditionViolation & c)
        // {
            // std::string expected("\nPrecondition violation!\ninverse(): matrix is not invertible.");
            // std::string message(c.what());
            // should(0 == expected.compare(message.substr(0,expected.size())));
        // }

        // // test pseudo-inverse
        // double data2[] = { 0.,  1.,  0.,  0.,  0.,
                           // 0.,  0.,  1.,  0.,  2.,
                           // 2.,  0.,  0.,  3.,  0. };
        // double refdata[] = {  0.0, 0.0, 0.15384615384615388,
                              // 1.0, 0.0, 0.0,
                              // 0.0, 0.2, 0.0,
                              // 0.0, 0.0, 0.23076923076923081,
                              // 0.0, 0.4, 0.0 };

        // Matrix m(3, 5, data2), piref(5, 3, refdata), pitref(transpose(piref));
        // Matrix pi = inverse(m);
        // shouldEqual(pi.shape(), Shape(5, 3));
        // shouldEqualSequenceTolerance(piref.data(), piref.data()+15, pi.data(), 1e-15);

        // Matrix pit = inverse(transpose(m));
        // shouldEqual(pit.shape(), Shape(3, 5));
        // shouldEqualSequenceTolerance(pitref.data(), pitref.data()+15, pit.data(), 1e-15);
    // }

    // void testSymmetricEigensystem()
    // {
        // double epsilon = 1e-8;
        // Matrix idref = identityMatrix<double>(size);

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_symmetric_matrix (size);
            // Matrix ew(size, 1);
            // Matrix ev(size, size);
            // should(symmetricEigensystem(a, ew, ev));
            // Matrix id = ev * transpose(ev);
            // shouldEqualSequenceTolerance(id.data(), id.data()+size*size, idref.data(), epsilon);
            // Matrix ae = ev * diagonalMatrix(ew) * transpose(ev);
            // shouldEqualSequenceTolerance(ae.data(), ae.data()+size*size, a.data(), epsilon);
        // }
    // }

    // void testSymmetricEigensystemAnalytic()
    // {
        // double epsilon = 1e-8;

        // int size = 2;
        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_symmetric_matrix (size);
            // Matrix ew(size, 1), ewref(size, 1);
            // Matrix ev(size, size);
            // symmetricEigensystem(a, ewref, ev);
            // symmetric2x2Eigenvalues(
                // a(0,0), a(0,1),
                // a(1,1),
                // &ew(0,0), &ew(1,0));
            // shouldEqualSequenceTolerance(ew.data(), ew.data()+size, ewref.data(), epsilon);
        // }

        // size = 3;
        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_symmetric_matrix (size);
            // Matrix ew(size, 1), ewref(size, 1);
            // Matrix ev(size, size);
            // symmetricEigensystem(a, ewref, ev);
            // symmetric3x3Eigenvalues<double>(
                // a(0,0), a(0,1), a(0,2),
                // a(1,1), a(1,2),
                // a(2,2),
                // &ew(0,0), &ew(1,0), &ew(2,0));
            // shouldEqualSequenceTolerance(ew.data(), ew.data()+size, ewref.data(), epsilon);
        // }
    // }

    // void testNonsymmetricEigensystem()
    // {
        // double epsilon = 1e-8;
        // Matrix idref = identityMatrix<double>(size);

        // for(unsigned int i = 0; i < iterations; ++i)
        // {
            // Matrix a = random_matrix (size, size);
            // Matrix<std::complex<double> > ew(size, 1);
            // Matrix ev(size, size);
            // should(nonsymmetricEigensystem(a, ew, ev));

            // Matrix ewm(size, size);
            // for(unsigned int k = 0; k < size; k++)
            // {
                // ewm(k, k) = ew(k, 0).real();
                // if(ew(k, 0).imag() > 0.0)
                // {
                    // ewm(k, k+1) = ew(k, 0).imag();
                // }
                // else if(ew(k, 0).imag() < 0.0)
                // {
                    // ewm(k, k-1) = ew(k, 0).imag();
                // }
            // }
            // Matrix ae = ev * ewm * inverse(ev);
            // shouldEqualSequenceTolerance(ae.data(), ae.data()+size*size, a.data(), epsilon);
        // }
    // }

    // void testDeterminant()
    // {
        // double ds2[] = {1, 2, 2, 1};
        // double dns2[] = {1, 2, 3, 1};
        // Matrix ms2(Shape(2,2), ds2);
        // Matrix mns2(Shape(2,2), dns2);
        // double eps = 1e-12;
        // shouldEqualTolerance(determinant(ms2), -3.0, eps);
        // shouldEqualTolerance(determinant(mns2), -5.0, eps);
        // shouldEqualTolerance(logDeterminant(transpose(ms2)*ms2), std::log(9.0), eps);
        // shouldEqualTolerance(logDeterminant(transpose(mns2)*mns2), std::log(25.0), eps);

        // double ds3[] = {1, 2, 3, 2, 3, 1, 3, 1, 2};
        // double dns3[] = {1, 2, 3, 5, 3, 1, 3, 1, 2};
        // Matrix ms3(Shape(3,3), ds3);
        // Matrix mns3(Shape(3,3), dns3);
        // shouldEqualTolerance(determinant(ms3), -18.0, eps);
        // shouldEqualTolerance(determinant(mns3), -21.0, eps);
        // shouldEqualTolerance(determinant(transpose(ms3)*ms3, "Cholesky"), 324.0, eps);
        // shouldEqualTolerance(determinant(transpose(mns3)*mns3, "Cholesky"), 441.0, eps);
        // shouldEqualTolerance(logDeterminant(transpose(ms3)*ms3), std::log(324.0), eps);
        // shouldEqualTolerance(logDeterminant(transpose(mns3)*mns3), std::log(441.0), eps);
    // }

    // void testSVD()
    // {
        // unsigned int m = 6, n = 4;
        // Matrix a(m, n);
        // for(unsigned int i1= 0; i1 < m; i1++)
            // for(unsigned int i2= 0; i2 < n; i2++)
                // a(i1, i2)= random_double();
        // Matrix u(m, n);
        // Matrix v(n, n);
        // Matrix S(n, 1);

        // unsigned int rank = singularValueDecomposition(a, u, S, v);
        // shouldEqual(rank, n);

        // double eps = 1e-11;

        // shouldEqualToleranceMessage(norm(a-u*diagonalMatrix(S)*transpose(v)), 0.0, eps, VIGRA_TOLERANCE_MESSAGE);
        // shouldEqualToleranceMessage(norm(identityMatrix<double>(4) - transpose(u)*u), 0.0, eps, VIGRA_TOLERANCE_MESSAGE);
        // shouldEqualToleranceMessage(norm(identityMatrix<double>(4) - transpose(v)*v), 0.0, eps, VIGRA_TOLERANCE_MESSAGE);
        // shouldEqualToleranceMessage(norm(identityMatrix<double>(4) - v*transpose(v)), 0.0, eps, VIGRA_TOLERANCE_MESSAGE);
    // }
};

// struct RandomTest
// {
    // void testTT800()
    // {
        // const unsigned int n = 50;
        // unsigned int iref[n] = {
            // 3169973338U, 2724982910U,  347012937U, 1735893326U, 2282497071U,
            // 3975116866U,   62755666U,  500522132U,  129776071U, 1978109378U,
            // 4040131704U, 3800592193U, 3057303977U, 1468369496U,  370579849U,
            // 3630178833U,   51910867U,  819270944U,  476180518U,  190380673U,
            // 1370447020U, 1620916304U,  663482756U, 1354889312U, 4000276916U,
             // 868393086U, 1441698743U, 1086138563U, 1899869374U, 3717419747U,
            // 2455034041U, 2617437696U, 1595651084U, 4148285605U, 1860328467U,
             // 928897371U,  263340857U, 4091726170U, 2359987311U, 1669697327U,
            // 1882626857U, 1635656338U,  897501559U, 3233276032U,  373770970U,
            // 2950632840U, 2706386845U, 3294066568U, 3819538748U, 1902519841U };

        // RandomTT800 random;
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(random(), iref[k]);

        // double fref[n] = {
              // 0.738067,   0.634460,   0.080795,   0.404169,   0.531435,
              // 0.925529,   0.014611,   0.116537,   0.030216,   0.460564,
              // 0.940666,   0.884894,   0.711834,   0.341881,   0.086282,
              // 0.845217,   0.012086,   0.190751,   0.110869,   0.044326,
              // 0.319082,   0.377399,   0.154479,   0.315460,   0.931387,
              // 0.202189,   0.335672,   0.252886,   0.442348,   0.865529,
              // 0.571607,   0.609420,   0.371516,   0.965848,   0.433141,
              // 0.216276,   0.061314,   0.952679,   0.549477,   0.388757,
              // 0.438333,   0.380831,   0.208966,   0.752806,   0.087025,
              // 0.686998,   0.630130,   0.766960,   0.889306,   0.442965 };
        // RandomTT800 randomf;
        // for(unsigned int k=0; k<n; ++k)
            // should(abs(randomf.uniform() - fref[k]) < 2e-6);

        // RandomTT800 randomr(RandomSeed);
    // }

    // void testMT19937()
    // {
        // const unsigned int n = 20, skip = 960, ilen = 4;
        // unsigned int first[n] = {
             // 956529277U, 3842322136U, 3319553134U, 1843186657U, 2704993644U,
             // 595827513U,  938518626U, 1676224337U, 3221315650U, 1819026461U,
            // 2401778706U, 2494028885U,  767405145U, 1590064561U, 2766888951U,
            // 3951114980U, 2568046436U, 2550998890U, 2642089177U,  568249289U };
        // unsigned int last[n] = {
            // 2396869032U, 1982500200U, 2649478910U,  839934727U, 3814542520U,
             // 918389387U,  995030736U, 2017568170U, 2621335422U, 1020082601U,
              // 24244213U, 2575242697U, 3941971804U,  922591409U, 2851763435U,
            // 2055641408U, 3695291669U, 2040276077U, 4118847636U, 3528766079U };

        // RandomMT19937 random(0xDEADBEEF);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(random(), first[k]);
        // for(unsigned int k=0; k<skip; ++k)
            // random();
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(random(), last[k]);

        // for(unsigned int k=0; k<skip; ++k)
            // should(random.uniformInt(31) < 31);

        // random.seed(0xDEADBEEF);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(random(), first[k]);
        // for(unsigned int k=0; k<skip; ++k)
            // random();
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(random(), last[k]);

        // unsigned int firsta[n] = {
            // 1067595299U,  955945823U,  477289528U, 4107218783U, 4228976476U,
            // 3344332714U, 3355579695U,  227628506U,  810200273U, 2591290167U,
            // 2560260675U, 3242736208U,  646746669U, 1479517882U, 4245472273U,
            // 1143372638U, 3863670494U, 3221021970U, 1773610557U, 1138697238U };
        // unsigned int lasta[n] = {
             // 123599888U,  472658308U, 1053598179U, 1012713758U, 3481064843U,
            // 3759461013U, 3981457956U, 3830587662U, 1877191791U, 3650996736U,
             // 988064871U, 3515461600U, 4089077232U, 2225147448U, 1249609188U,
            // 2643151863U, 3896204135U, 2416995901U, 1397735321U, 3460025646U };

        // unsigned int init[ilen] = {0x123, 0x234, 0x345, 0x456};
        // RandomMT19937 randoma(init, ilen);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(randoma(), firsta[k]);
        // for(unsigned int k=0; k<skip; ++k)
            // randoma();
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(randoma(), lasta[k]);

        // double ref53[n] = {
            // 0.76275444, 0.98670464, 0.27933125, 0.94218739, 0.78842173,
            // 0.92179002, 0.54534773, 0.38107717, 0.65286910, 0.22765212,
            // 0.74557914, 0.54708246, 0.42043117, 0.19189126, 0.70259889,
            // 0.77408120, 0.04605807, 0.69398269, 0.61711170, 0.10133577};
        // for(unsigned int k=0; k<n; ++k)
            // should(abs(randoma.uniform53()-ref53[k]) < 2e-8);

        // randoma.seed(init, ilen);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(randoma(), firsta[k]);
        // for(unsigned int k=0; k<skip; ++k)
            // randoma();
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(randoma(), lasta[k]);
    // }

    // void testRandomFunctors()
    // {
        // const unsigned int n = 50;
        // unsigned int iref[n] = {
            // 3169973338U, 2724982910U,  347012937U, 1735893326U, 2282497071U,
            // 3975116866U,   62755666U,  500522132U,  129776071U, 1978109378U,
            // 4040131704U, 3800592193U, 3057303977U, 1468369496U,  370579849U,
            // 3630178833U,   51910867U,  819270944U,  476180518U,  190380673U,
            // 1370447020U, 1620916304U,  663482756U, 1354889312U, 4000276916U,
             // 868393086U, 1441698743U, 1086138563U, 1899869374U, 3717419747U,
            // 2455034041U, 2617437696U, 1595651084U, 4148285605U, 1860328467U,
             // 928897371U,  263340857U, 4091726170U, 2359987311U, 1669697327U,
            // 1882626857U, 1635656338U,  897501559U, 3233276032U,  373770970U,
            // 2950632840U, 2706386845U, 3294066568U, 3819538748U, 1902519841U };
        // double fref[n] = {
              // 0.738067,   0.634460,   0.080795,   0.404169,   0.531435,
              // 0.925529,   0.014611,   0.116537,   0.030216,   0.460564,
              // 0.940666,   0.884894,   0.711834,   0.341881,   0.086282,
              // 0.845217,   0.012086,   0.190751,   0.110869,   0.044326,
              // 0.319082,   0.377399,   0.154479,   0.315460,   0.931387,
              // 0.202189,   0.335672,   0.252886,   0.442348,   0.865529,
              // 0.571607,   0.609420,   0.371516,   0.965848,   0.433141,
              // 0.216276,   0.061314,   0.952679,   0.549477,   0.388757,
              // 0.438333,   0.380831,   0.208966,   0.752806,   0.087025,
              // 0.686998,   0.630130,   0.766960,   0.889306,   0.442965 };
        // double nref[n] = {
            // 1.35298, 0.764158, -0.757076, -0.173069, 0.0586711,
            // 0.794212, -0.483372, -0.0405762, 1.27956, -0.955101,
            // -1.5062, -1.02069, -0.871562, -0.465495, -0.799888,
            // -1.20286, -0.170944, 1.08383, 1.26832, 1.93807,
            // -0.098183, 0.355986, -0.336965, -1.42996, 0.966012,
            // -2.17195, -1.05422, -2.03724, -0.769992, 0.668851,
            // -0.570259, 0.258217, 0.632492, 1.29755, 0.96869,
            // -0.141918, -0.836236, -0.62337, 0.116509, -0.0314471,
            // 0.402451, -1.20504, -0.140861, -0.0765263, 1.06057,
            // 2.57671, 0.0299117, 0.471425, 1.59464, 1.37346};

        // RandomTT800 random1;
        // UniformRandomFunctor<RandomTT800> f1(random1);
        // for(unsigned int k=0; k<n; ++k)
            // should(abs(f1() - fref[k]) < 2e-6);

        // RandomTT800 random2;
        // UniformIntRandomFunctor<RandomTT800> f2(4, 34, random2, true);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(f2(), iref[k] % 31 + 4);

        // RandomTT800 random3;
        // UniformIntRandomFunctor<RandomTT800> f3(random3);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqual(f3(32), iref[k] % 32);

        // RandomTT800 random4;
        // NormalRandomFunctor<RandomTT800> f4(random4);
        // for(unsigned int k=0; k<n; ++k)
            // shouldEqualTolerance(f4(), nref[k], 1e-5);
    // }
// };


struct MatrixTestSuite
: public test_suite
{
    MatrixTestSuite()
    : test_suite("MatrixTestSuite")
    {
        // add( testCase(&MatrixTest::testOStreamShifting));
        add( testCase(&MatrixTest::testMatrix));
        add( testCase(&MatrixTest::testArgMinMax));
        // add( testCase(&MatrixTest::testColumnAndRowStatistics));
        // add( testCase(&MatrixTest::testColumnAndRowPreparation));
        // add( testCase(&MatrixTest::testCholesky));
        // add( testCase(&MatrixTest::testQR));
        // add( testCase(&MatrixTest::testLinearSolve));
        // add( testCase(&MatrixTest::testUnderdetermined));
        // add( testCase(&MatrixTest::testOverdetermined));
        // add( testCase(&MatrixTest::testIncrementalLinearSolve));
        // add( testCase(&MatrixTest::testInverse));
        // add( testCase(&MatrixTest::testSymmetricEigensystem));
        // add( testCase(&MatrixTest::testNonsymmetricEigensystem));
        // add( testCase(&MatrixTest::testSymmetricEigensystemAnalytic));
        // add( testCase(&MatrixTest::testDeterminant));
        // add( testCase(&MatrixTest::testSVD));

        // add( testCase(&RandomTest::testTT800));
        // add( testCase(&RandomTest::testMT19937));
        // add( testCase(&RandomTest::testRandomFunctors));
    }
};

int main(int argc, char ** argv)
{
    MatrixTestSuite test;

    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cerr << test.report() << std::endl;

    return (failed != 0);
}
