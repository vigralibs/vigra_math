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
#include <vigra2/unittest.hxx>
#include <vigra2/math/quaternion.hxx>

using namespace vigra;

struct QuaternionTest
{
    typedef Quaternion<double> Q;
    typedef Q::Vector V;

    void testContents()
    {
        Q q(1.0, 2.0, 3.0, 4.0), q0, q1(-1.0), q2(q), q3(q.w(), q.v());

        shouldEqual(q.w(), 1.0);
        shouldEqual(q.v(), V(2.0, 3.0, 4.0));
        shouldEqual(q0.w(), 0.0);
        shouldEqual(q0.v(), V(0.0, 0.0, 0.0));
        shouldEqual(q1.w(), -1.0);
        shouldEqual(q1.v(), V(0.0, 0.0, 0.0));
        shouldEqual(q2.w(), 1.0);
        shouldEqual(q2.v(), V(2.0, 3.0, 4.0));
        shouldEqual(q3.w(), 1.0);
        shouldEqual(q3.v(), V(2.0, 3.0, 4.0));

        shouldEqual(q[0], 1.0);
        shouldEqual(q[1], 2.0);
        shouldEqual(q[2], 3.0);
        shouldEqual(q[3], 4.0);
        shouldEqual(q.x(), 2.0);
        shouldEqual(q.y(), 3.0);
        shouldEqual(q.z(), 4.0);

        should(q == q2);
        should(q1 != q2);

        q2 = q1;
        shouldEqual(q2.w(), -1.0);
        shouldEqual(q2.v(), V(0.0, 0.0, 0.0));

        should(q != q2);
        should(q1 == q2);

        q3 = 10.0;
        shouldEqual(q3.w(), 10.0);
        shouldEqual(q3.v(), V(0.0, 0.0, 0.0));

        q2.setW(-2.0);
        shouldEqual(q2.w(), -2.0);
        shouldEqual(q2.v(), V(0.0, 0.0, 0.0));

        q2.setV(V(5.0, 6.0, 7.0));
        shouldEqual(q2.w(), -2.0);
        shouldEqual(q2.v(), V(5.0, 6.0, 7.0));

        q3.setV(5.0, 6.0, 7.0);
        shouldEqual(q3.w(), 10.0);
        shouldEqual(q3.v(), V(5.0, 6.0, 7.0));

        q3.setX(2.0);
        q3.setY(3.0);
        q3.setZ(4.0);
        shouldEqual(q3.w(), 10.0);
        shouldEqual(q3.v(), V(2.0, 3.0, 4.0));

        shouldEqual(squaredNorm(q), 30.0);
        shouldEqualTolerance(norm(q), sqrt(30.0), 1e-15);
        shouldEqual(norm(q), abs(q));
    }

    void testStreamIO()
    {
        std::ostringstream out;
        Q q(1.0, 2.0, 3.0, 4.0);

        out << q;
        shouldEqual(out.str(), "1 2 3 4");

        std::istringstream in;
        in.str("10 11 12 13");
        in >> q;
        shouldEqual(q, Q(10.0, 11.0, 12.0, 13.0));
    }

    void testOperators()
    {
        Q q(1.0, 2.0, 3.0, 4.0);

        shouldEqual(+q, q);
        shouldEqual(-q, Q(-1,-2,-3,-4));

        shouldEqual(q+q, Q(2,4,6,8));
        shouldEqual(q+2.0, Q(3,2,3,4));
        shouldEqual(2.0+q, Q(3,2,3,4));

        shouldEqual(Q(2,4,6,8) - q, q);
        shouldEqual(q-2.0, Q(-1,2,3,4));
        shouldEqual(2.0-q, Q(1,-2,-3,-4));

        shouldEqual(Q(1,0,0,0)*Q(1,0,0,0), Q(1,0,0,0));
        shouldEqual(Q(0,1,0,0)*Q(0,1,0,0), Q(-1,0,0,0));
        shouldEqual(Q(0,0,1,0)*Q(0,0,1,0), Q(-1,0,0,0));
        shouldEqual(Q(0,0,0,1)*Q(0,0,0,1), Q(-1,0,0,0));

        shouldEqual(Q(0,1,0,0)*Q(0,0,1,0), Q(0,0,0,1));
        shouldEqual(Q(0,0,1,0)*Q(0,1,0,0), Q(0,0,0,-1));
        shouldEqual(Q(0,0,1,0)*Q(0,0,0,1), Q(0,1,0,0));
        shouldEqual(Q(0,0,0,1)*Q(0,0,1,0), Q(0,-1,0,0));
        shouldEqual(Q(0,0,0,1)*Q(0,1,0,0), Q(0,0,1,0));
        shouldEqual(Q(0,1,0,0)*Q(0,0,0,1), Q(0,0,-1,0));

        shouldEqual(q*q, Q(-28,4,6,8));
        shouldEqual(q*2.0, Q(2,4,6,8));
        shouldEqual(2.0*q, Q(2,4,6,8));

        Q q1 = q / q;
        shouldEqualTolerance(q1[0], 1.0, 1e-16);
        shouldEqualTolerance(q1[1], 0.0, 1e-16);
        shouldEqualTolerance(q1[2], 0.0, 1e-16);
        shouldEqualTolerance(q1[3], 0.0, 1e-16);
        shouldEqual(Q(2,4,6,8)/2.0, q);
        shouldEqual(60.0/q, Q(2,-4,-6,-8));

        shouldEqualTolerance(norm(q / norm(q)), 1.0, 1e-15);
    }

    void testRotation()
    {
        typedef TinyArray<double, 3, 3> Matrix;

         Q q(1.0, 2.0, 3.0, 4.0);
         q /= norm(q);

         Matrix ref = { -2.0 / 3.0,  0.4 / 3.0, 2.2 / 3.0,
                         2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0,
                         1.0 / 3.0,  2.8 / 3.0, 0.4 / 3.0 };

         Matrix r;
         q.fillRotationMatrix(r);
         shouldEqualSequenceTolerance(r.begin(), r.end(), ref.begin(), 1e-15);

         Q q1 = Q::createRotation(M_PI/2.0, V(1., 0., 0.));
         Q q2 = Q::createRotation(M_PI/2.0, V(0., 1., 0.));
         Q q3 = Q::createRotation(M_PI/2.0, V(0., 0., 1.));
         Q q4 = q3*(-q1)*q2*q1;

         shouldEqualTolerance(norm(q4), 1.0, 1e-15);
         shouldEqualTolerance(q4[0], 0.0, 1e-15);
    }
};

struct QuaternionTestSuite
: public test_suite
{
    QuaternionTestSuite()
    : test_suite("QuaternionTestSuite")
    {
        add( testCase(&QuaternionTest::testContents));
        add( testCase(&QuaternionTest::testStreamIO));
        add( testCase(&QuaternionTest::testOperators));
        add( testCase(&QuaternionTest::testRotation));
   }
};

int main(int argc, char ** argv)
{
    QuaternionTestSuite test;

    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cerr << test.report() << std::endl;

    return (failed != 0);
}
