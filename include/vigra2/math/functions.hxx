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

#ifndef VIGRA2_MATH_FUNCTIONS_HXX
#define VIGRA2_MATH_FUNCTIONS_HXX

#include <vigra2/mathutil.hxx>

namespace vigra {

/** \addtogroup MathFunctions
*/
//@{
    /** \brief Compute the eigenvalues of a 2x2 real symmetric matrix.

        This uses the analytical eigenvalue formula
        \f[
           \lambda_{1,2} = \frac{1}{2}\left(a_{00} + a_{11} \pm \sqrt{(a_{00} - a_{11})^2 + 4 a_{01}^2}\right)
        \f]

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T>
void symmetric2x2Eigenvalues(T a00, T a01, T a11, T * r0, T * r1)
{
    double d  = hypot(a00 - a11, 2.0*a01);
    *r0 = static_cast<T>(0.5*(a00 + a11 + d));
    *r1 = static_cast<T>(0.5*(a00 + a11 - d));
    if(*r0 < *r1)
        std::swap(*r0, *r1);
}

    /** \brief Compute the eigenvalues of a 3x3 real symmetric matrix.

        This uses a numerically stable version of the analytical eigenvalue formula according to
        <p>
        David Eberly: <a href="http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf">
        <em>"Eigensystems for 3 × 3 Symmetric Matrices (Revisited)"</em></a>, Geometric Tools Documentation, 2006

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class T>
void symmetric3x3Eigenvalues(T a00, T a01, T a02, T a11, T a12, T a22,
                             T * r0, T * r1, T * r2)
{
    double inv3 = 1.0 / 3.0, root3 = std::sqrt(3.0);

    double c0 = a00*a11*a22 + 2.0*a01*a02*a12 - a00*a12*a12 - a11*a02*a02 - a22*a01*a01;
    double c1 = a00*a11 - a01*a01 + a00*a22 - a02*a02 + a11*a22 - a12*a12;
    double c2 = a00 + a11 + a22;
    double c2Div3 = c2*inv3;
    double aDiv3 = (c1 - c2*c2Div3)*inv3;
    if (aDiv3 > 0.0)
        aDiv3 = 0.0;
    double mbDiv2 = 0.5*(c0 + c2Div3*(2.0*c2Div3*c2Div3 - c1));
    double q = mbDiv2*mbDiv2 + aDiv3*aDiv3*aDiv3;
    if (q > 0.0)
        q = 0.0;
    double magnitude = std::sqrt(-aDiv3);
    double angle = std::atan2(std::sqrt(-q),mbDiv2)*inv3;
    double cs = std::cos(angle);
    double sn = std::sin(angle);
    *r0 = static_cast<T>(c2Div3 + 2.0*magnitude*cs);
    *r1 = static_cast<T>(c2Div3 - magnitude*(cs + root3*sn));
    *r2 = static_cast<T>(c2Div3 - magnitude*(cs - root3*sn));
    if(*r0 < *r1)
        std::swap(*r0, *r1);
    if(*r0 < *r2)
        std::swap(*r0, *r2);
    if(*r1 < *r2)
        std::swap(*r1, *r2);
}

namespace detail {

template <class T>
T ellipticRD(T x, T y, T z)
{
    double f = 1.0, s = 0.0, X, Y, Z, m;
    for(;;)
    {
        m = (x + y + 3.0*z) / 5.0;
        X = 1.0 - x/m;
        Y = 1.0 - y/m;
        Z = 1.0 - z/m;
        if(std::max(std::max(VIGRA_CSTD::fabs(X), VIGRA_CSTD::fabs(Y)), VIGRA_CSTD::fabs(Z)) < 0.01)
            break;
        double l = VIGRA_CSTD::sqrt(x*y) + VIGRA_CSTD::sqrt(x*z) + VIGRA_CSTD::sqrt(y*z);
        s += f / (VIGRA_CSTD::sqrt(z)*(z + l));
        f /= 4.0;
        x = (x + l)/4.0;
        y = (y + l)/4.0;
        z = (z + l)/4.0;
    }
    double a = X*Y;
    double b = sq(Z);
    double c = a - b;
    double d = a - 6.0*b;
    double e = d + 2.0*c;
    return 3.0*s + f*(1.0+d*(-3.0/14.0+d*9.0/88.0-Z*e*4.5/26.0)
                      +Z*(e/6.0+Z*(-c*9.0/22.0+a*Z*3.0/26.0))) / VIGRA_CSTD::pow(m,1.5);
}

template <class T>
T ellipticRF(T x, T y, T z)
{
    double X, Y, Z, m;
    for(;;)
    {
        m = (x + y + z) / 3.0;
        X = 1.0 - x/m;
        Y = 1.0 - y/m;
        Z = 1.0 - z/m;
        if(std::max(std::max(VIGRA_CSTD::fabs(X), VIGRA_CSTD::fabs(Y)), VIGRA_CSTD::fabs(Z)) < 0.01)
            break;
        double l = VIGRA_CSTD::sqrt(x*y) + VIGRA_CSTD::sqrt(x*z) + VIGRA_CSTD::sqrt(y*z);
        x = (x + l)/4.0;
        y = (y + l)/4.0;
        z = (z + l)/4.0;
    }
    double d = X*Y - sq(Z);
    double p = X*Y*Z;
    return (1.0 - d/10.0 + p/14.0 + sq(d)/24.0 - d*p*3.0/44.0) / VIGRA_CSTD::sqrt(m);
}

} // namespace detail

    /** \brief The incomplete elliptic integral of the first kind.

        This function computes

        \f[
             \mbox{F}(x, k) = \int_0^x \frac{1}{\sqrt{1 - k^2 \sin(t)^2}} dt
        \f]

        according to the algorithm given in Press et al. "Numerical Recipes".

        Note: In some libraries (e.g. Mathematica), the second parameter of the elliptic integral
        functions must be k^2 rather than k. Check the documentation when results disagree!

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double ellipticIntegralF(double x, double k)
{
    double c2 = sq(VIGRA_CSTD::cos(x));
    double s = VIGRA_CSTD::sin(x);
    return s*detail::ellipticRF(c2, 1.0 - sq(k*s), 1.0);
}

    /** \brief The incomplete elliptic integral of the second kind.

        This function computes

        \f[
            \mbox{E}(x, k) = \int_0^x \sqrt{1 - k^2 \sin(t)^2} dt
        \f]

        according to the algorithm given in Press et al. "Numerical Recipes". The
        complete elliptic integral of the second kind is simply <tt>ellipticIntegralE(M_PI/2, k)</TT>.

        Note: In some libraries (e.g. Mathematica), the second parameter of the elliptic integral
        functions must be k^2 rather than k. Check the documentation when results disagree!

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double ellipticIntegralE(double x, double k)
{
    double c2 = sq(VIGRA_CSTD::cos(x));
    double s = VIGRA_CSTD::sin(x);
    k = sq(k*s);
    return s*(detail::ellipticRF(c2, 1.0-k, 1.0) - k/3.0*detail::ellipticRD(c2, 1.0-k, 1.0));
}

namespace detail {

template <class T>
double noncentralChi2CDFApprox(unsigned int degreesOfFreedom, T noncentrality, T arg)
{
    double a = degreesOfFreedom + noncentrality;
    double b = (a + noncentrality) / sq(a);
    double t = (VIGRA_CSTD::pow((double)arg / a, 1.0/3.0) - (1.0 - 2.0 / 9.0 * b)) / VIGRA_CSTD::sqrt(2.0 / 9.0 * b);
    return 0.5*(1.0 + erf(t/VIGRA_CSTD::sqrt(2.0)));
}

template <class T>
void noncentralChi2OneIteration(T arg, T & lans, T & dans, T & pans, unsigned int & j)
{
    double tol = -50.0;
    if(lans < tol)
    {
        lans = lans + VIGRA_CSTD::log(arg / j);
        dans = VIGRA_CSTD::exp(lans);
    }
    else
    {
        dans = dans * arg / j;
    }
    pans = pans - dans;
    j += 2;
}

template <class T>
std::pair<double, double> noncentralChi2CDF(unsigned int degreesOfFreedom, T noncentrality, T arg, T eps)
{
    vigra_precondition(noncentrality >= 0.0 && arg >= 0.0 && eps > 0.0,
        "noncentralChi2P(): parameters must be positive.");
    if (arg == 0.0 && degreesOfFreedom > 0)
        return std::make_pair(0.0, 0.0);

    // Determine initial values
    double b1 = 0.5 * noncentrality,
           ao = VIGRA_CSTD::exp(-b1),
           eps2 = eps / ao,
           lnrtpi2 = 0.22579135264473,
           probability, density, lans, dans, pans, sum, am, hold;
    unsigned int maxit = 500,
        i, m;
    if(degreesOfFreedom % 2)
    {
        i = 1;
        lans = -0.5 * (arg + VIGRA_CSTD::log(arg)) - lnrtpi2;
        dans = VIGRA_CSTD::exp(lans);
        pans = erf(VIGRA_CSTD::sqrt(arg/2.0));
    }
    else
    {
        i = 2;
        lans = -0.5 * arg;
        dans = VIGRA_CSTD::exp(lans);
        pans = 1.0 - dans;
    }

    // Evaluate first term
    if(degreesOfFreedom == 0)
    {
        m = 1;
        degreesOfFreedom = 2;
        am = b1;
        sum = 1.0 / ao - 1.0 - am;
        density = am * dans;
        probability = 1.0 + am * pans;
    }
    else
    {
        m = 0;
        degreesOfFreedom = degreesOfFreedom - 1;
        am = 1.0;
        sum = 1.0 / ao - 1.0;
        while(i < degreesOfFreedom)
            detail::noncentralChi2OneIteration(arg, lans, dans, pans, i);
        degreesOfFreedom = degreesOfFreedom + 1;
        density = dans;
        probability = pans;
    }
    // Evaluate successive terms of the expansion
    for(++m; m<maxit; ++m)
    {
        am = b1 * am / m;
        detail::noncentralChi2OneIteration(arg, lans, dans, pans, degreesOfFreedom);
        sum = sum - am;
        density = density + am * dans;
        hold = am * pans;
        probability = probability + hold;
        if((pans * sum < eps2) && (hold < eps2))
            break; // converged
    }
    if(m == maxit)
        vigra_fail("noncentralChi2P(): no convergence.");
    return std::make_pair(0.5 * ao * density, std::min(1.0, std::max(0.0, ao * probability)));
}

} // namespace detail

    /** \brief Chi square distribution.

        Computes the density of a chi square distribution with \a degreesOfFreedom
        and tolerance \a accuracy at the given argument \a arg
        by calling <tt>noncentralChi2(degreesOfFreedom, 0.0, arg, accuracy)</tt>.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double chi2(unsigned int degreesOfFreedom, double arg, double accuracy = 1e-7)
{
    return detail::noncentralChi2CDF(degreesOfFreedom, 0.0, arg, accuracy).first;
}

    /** \brief Cumulative chi square distribution.

        Computes the cumulative density of a chi square distribution with \a degreesOfFreedom
        and tolerance \a accuracy at the given argument \a arg, i.e. the probability that
        a random number drawn from the distribution is below \a arg
        by calling <tt>noncentralChi2CDF(degreesOfFreedom, 0.0, arg, accuracy)</tt>.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double chi2CDF(unsigned int degreesOfFreedom, double arg, double accuracy = 1e-7)
{
    return detail::noncentralChi2CDF(degreesOfFreedom, 0.0, arg, accuracy).second;
}

    /** \brief Non-central chi square distribution.

        Computes the density of a non-central chi square distribution with \a degreesOfFreedom,
        noncentrality parameter \a noncentrality and tolerance \a accuracy at the given argument
        \a arg. It uses Algorithm AS 231 from Appl. Statist. (1987) Vol.36, No.3 (code ported from
        http://lib.stat.cmu.edu/apstat/231). The algorithm has linear complexity in the number of
        degrees of freedom.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double noncentralChi2(unsigned int degreesOfFreedom,
              double noncentrality, double arg, double accuracy = 1e-7)
{
    return detail::noncentralChi2CDF(degreesOfFreedom, noncentrality, arg, accuracy).first;
}

    /** \brief Cumulative non-central chi square distribution.

        Computes the cumulative density of a chi square distribution with \a degreesOfFreedom,
        noncentrality parameter \a noncentrality and tolerance \a accuracy at the given argument
        \a arg, i.e. the probability that a random number drawn from the distribution is below \a arg
        It uses Algorithm AS 231 from Appl. Statist. (1987) Vol.36, No.3 (code ported from
        http://lib.stat.cmu.edu/apstat/231). The algorithm has linear complexity in the number of
        degrees of freedom (see noncentralChi2CDFApprox() for a constant-time algorithm).

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double noncentralChi2CDF(unsigned int degreesOfFreedom,
              double noncentrality, double arg, double accuracy = 1e-7)
{
    return detail::noncentralChi2CDF(degreesOfFreedom, noncentrality, arg, accuracy).second;
}

    /** \brief Cumulative non-central chi square distribution (approximate).

        Computes approximate values of the cumulative density of a chi square distribution with \a degreesOfFreedom,
        and noncentrality parameter \a noncentrality at the given argument
        \a arg, i.e. the probability that a random number drawn from the distribution is below \a arg
        It uses the approximate transform into a normal distribution due to Wilson and Hilferty
        (see Abramovitz, Stegun: "Handbook of Mathematical Functions", formula 26.3.32).
        The algorithm's running time is independent of the inputs, i.e. is should be used
        when noncentralChi2CDF() is too slow, and approximate values are sufficient. The accuracy is only
        about 0.1 for few degrees of freedom, but reaches about 0.001 above dof = 5.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
inline double noncentralChi2CDFApprox(unsigned int degreesOfFreedom, double noncentrality, double arg)
{
    return detail::noncentralChi2CDFApprox(degreesOfFreedom, noncentrality, arg);
}

namespace detail  {

// computes (l+m)! / (l-m)!
// l and m must be positive
template <class T>
T facLM(T l, T m)
{
    T tmp = NumericTraits<T>::one();
    for(T f = l-m+1; f <= l+m; ++f)
        tmp *= f;
    return tmp;
}

} // namespace detail

    /** \brief Associated Legendre polynomial.

        Computes the value of the associated Legendre polynomial of order <tt>l, m</tt>
        for argument <tt>x</tt>. <tt>x</tt> must be in the range <tt>[-1.0, 1.0]</tt>,
        otherwise an exception is thrown. The standard Legendre polynomials are the
        special case <tt>m == 0</tt>.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class REAL>
REAL legendre(unsigned int l, int m, REAL x)
{
    vigra_precondition(abs(x) <= 1.0, "legendre(): x must be in [-1.0, 1.0].");
    if (m < 0)
    {
        m = -m;
        REAL s = odd(m)
                   ? -1.0
                   :  1.0;
        return legendre(l,m,x) * s / detail::facLM<REAL>(l,m);
    }
    REAL result = 1.0;
    if (m > 0)
    {
        REAL r = std::sqrt( (1.0-x) * (1.0+x) );
        REAL f = 1.0;
        for (int i=1; i<=m; i++)
        {
            result *= (-f) * r;
            f += 2.0;
        }
    }
    if((int)l == m)
        return result;

    REAL result_1 = x * (2.0 * m + 1.0) * result;
    if((int)l == m+1)
        return result_1;
    REAL other = 0.0;
    for(unsigned int i = m+2; i <= l; ++i)
    {
        other = ( (2.0*i-1.0) * x * result_1 - (i+m-1.0)*result) / (i-m);
        result = result_1;
        result_1 = other;
    }
    return other;
}

    /** \brief \brief Legendre polynomial.

        Computes the value of the Legendre polynomial of order <tt>l</tt> for argument <tt>x</tt>.
        <tt>x</tt> must be in the range <tt>[-1.0, 1.0]</tt>, otherwise an exception is thrown.

        <b>\#include</b> \<vigra/mathutil.hxx\><br>
        Namespace: vigra
    */
template <class REAL>
REAL legendre(unsigned int l, REAL x)
{
    return legendre(l, 0, x);
}

//@}

} // namespace vigra

#endif /* VIGRA2_MATH_FUNCTIONS_HXX */
