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
#include <vigra2/math/functions.hxx>
// #include <vigra2/random.hxx>
// #include <vigra2/timing.hxx>

using namespace vigra;

struct FunctionsTest
{
    void testSpecialFunctions()
    {
        shouldEqualTolerance(ellipticIntegralE(M_PI / 2.0, 0.0), M_PI / 2.0, 1e-14);
        shouldEqualTolerance(ellipticIntegralF(0.3, 0.3), 0.30039919311549118, 1e-14);
        shouldEqualTolerance(ellipticIntegralE(0.3, 0.3), 0.29960175507025716, 1e-14);

        should(noncentralChi2CDFApprox(200, 0.0, 200.0) > 0.5);
        should(noncentralChi2CDFApprox(200, 0.0, 199.0) < 0.5);
        should(noncentralChi2CDF(200, 0.0, 200.0) > 0.5);
        should(noncentralChi2CDF(200, 0.0, 199.0) < 0.5);

        shouldEqualTolerance(noncentralChi2CDF(2, 2.0, 2.0), 0.34574583872316456, 1e-7);
        shouldEqualTolerance(noncentralChi2(2, 2.0, 2.0), 0.154254161276835, 1e-7);
        shouldEqualTolerance(noncentralChi2CDF(3, 2.0, 2.0), 0.22073308707450343, 1e-7);
        shouldEqualTolerance(noncentralChi2(3, 2.0, 2.0), 0.13846402271767755, 1e-7);
        shouldEqualTolerance(noncentralChi2CDFApprox(2, 2.0, 2.0), 0.34574583872316456, 1e-1);
        shouldEqualTolerance(noncentralChi2CDFApprox(3, 2.0, 2.0), 0.22073308707450343, 1e-1);

        double args[5] = {0.0, 1.0, 0.7, -0.7, -1.0};
        for(int i=0; i<5; ++i)
        {
            double x = args[i], x2 = x*x;
            shouldEqualTolerance(legendre(0, x), 1.0, 1e-15);
            shouldEqualTolerance(legendre(1, x), x, 1e-15);
            shouldEqualTolerance(legendre(2, x), 0.5*(3.0*x2-1.0), 1e-15);
            shouldEqualTolerance(legendre(3, x), 0.5*x*(5.0*x2-3.0), 1e-15);

            shouldEqualTolerance(legendre(0, 0, x), 1.0, 1e-15);
            shouldEqualTolerance(legendre(1, 0, x), x, 1e-15);
            shouldEqualTolerance(legendre(1, 1, x), -std::sqrt(1.0-x2), 1e-15);
            shouldEqualTolerance(legendre(2, 0, x), 0.5*(3.0*x2-1.0), 1e-15);
            shouldEqualTolerance(legendre(2, 1, x), -3.0*x*std::sqrt(1.0-x2), 1e-15);
            shouldEqualTolerance(legendre(2, 2, x), 3.0*(1.0-x2), 1e-15);
            shouldEqualTolerance(legendre(4, 2, x), 7.5*(7.0*x2-1.0)*(1.0-x2), 1e-15);
            shouldEqualTolerance(legendre(1, -1, x), -legendre(1, 1, x) / 2.0, 1e-15);
            shouldEqualTolerance(legendre(2, -1, x), -legendre(2, 1, x) / 6.0, 1e-15);
            shouldEqualTolerance(legendre(2, -2, x), legendre(2, 2, x) / 24.0, 1e-15);
        }
    }
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


struct FunctionsTestSuite
: public test_suite
{
    FunctionsTestSuite()
    : test_suite("FunctionsTest")
    {
        add( testCase(&FunctionsTest::testSpecialFunctions));

        // add( testCase(&RandomTest::testTT800));
        // add( testCase(&RandomTest::testMT19937));
        // add( testCase(&RandomTest::testRandomFunctors));
    }
};

int main(int argc, char ** argv)
{
  try
  {
    FunctionsTestSuite test;

    int failed = test.run(testsToBeExecuted(argc, argv));

    std::cerr << test.report() << std::endl;

    return (failed != 0);
  }
  catch(std::exception & e)
  {
    std::cerr << "Unexpected exception: " << e.what() << "\n";
    return 1;
  }
}
