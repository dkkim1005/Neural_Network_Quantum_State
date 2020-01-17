#pragma once

namespace MINRESQLP
{
template<typename DerivedOP, typename NumberType>
BaseInterface<DerivedOP, NumberType>::BaseInterface(const int n_, const NumberType * b_,
  const NumberTypeFloat<NumberType> shift_,
  const bool useMsolve_,
  const bool disable_,
  const int itnlim_,
  const NumberTypeFloat<NumberType> rtol_,
  const NumberTypeFloat<NumberType> maxxnorm_,
  const NumberTypeFloat<NumberType> trancond_,
  const NumberTypeFloat<NumberType> Acondlim_,
  const bool print_):
  // initialize list for inputs
  n(n_),
  b(b_, b_+n_),
  itnlim(itnlim_<0 ? 4*n_ : itnlim_),
  shift(shift_),
  useMsolve(useMsolve_),
  disable(disable_),
  rtol(rtol_),
  maxxnorm(maxxnorm_),
  trancond(trancond_),
  Acondlim(Acondlim_),
  print(print_),
  // initialize list for outputs
  x(n_),
  istop(0),
  itn(0),
  rnorm(0),
  Arnorm(0),
  xnorm(0),
  Acond(0) {}

template<typename DerivedOP, typename NumberType>
void BaseInterface<DerivedOP, NumberType>::Aprod(const int n, const NumberType *x, NumberType *y) const
{
  static_cast<const DerivedOP*>(this)->Aprod(n, x, y);
}

template<typename DerivedOP, typename NumberType>
void BaseInterface<DerivedOP, NumberType>::Msolve(const int n, const NumberType *x, NumberType *y) const
{
  static_cast<const DerivedOP*>(this)->Msolve(n, x, y);
}


template<typename DerivedOP, typename FloatType>
void RealSolver<DerivedOP, FloatType>::symortho_(const FloatType& a, const FloatType& b,
  FloatType &c, FloatType &s, FloatType &r) const
{
  FloatType t, abs_a = std::abs(a), abs_b = std::abs(b);
  if (abs_b <= eps_)
  {
    s = 0;
    r = abs_a;
    if (a == 0)
      c = 1;
    else
      c = a/abs_a;
  }
  else if (abs_a <= eps_)
  {
    c = 0;
    r = abs_b;
    s = b / abs_b;
  }
  else if (abs_b > abs_a)
  {
    t = a / b;
    s = (b / abs_b) / std::sqrt(1. + std::pow(t, 2));
    c = s*t;
    r = b/s;
  }
  else
  {
    t = b/a;
    c = (a/abs_a)/std::sqrt(1. + std::pow(t, 2));
    s = c*t;
    r = a/c;
  }
}

template<typename DerivedOP, typename FloatType>
FloatType RealSolver<DerivedOP, FloatType>::dnrm2_(const int n, const FloatType* x, const int incx) const
{
  int ix;
  FloatType ssq, absxi, norm, scale;
  if (n<1 || incx < 1)
    norm = 0;
  else if (n==1)
    norm = std::abs(x[0]);
  else
  {
    scale = 0, ssq = 1;
    for (ix=0; ix<(1+(n-1)*incx); ix += incx)
    {
      if (x[ix] != 0)
      {
        absxi = std::abs(x[ix]);
        if (scale < absxi)
        {
          ssq = 1. + ssq*std::pow(scale/absxi, 2);
          scale = absxi;
        }
        else
          ssq += std::pow(absxi/scale, 2);
      }
    }
    norm = scale*std::sqrt(ssq);
  }
  return norm;
}

template<typename DerivedOP, typename FloatType>
void RealSolver<DerivedOP, FloatType>::printstate_(const int iter, const FloatType x1, const FloatType xnorm,
  const FloatType rnorm, const FloatType Arnorm, const FloatType relres,
  const FloatType relAres, const FloatType Anorm, const FloatType Acond) const
{
  std::cout << std::setw(7) << "iter "
            << std::setw(14) << "x[0] "
            << std::setw(14) << "xnorm "
            << std::setw(14) << "rnorm "
            << std::setw(14) << "Arnorm "
            << std::setw(14) << "Compatible "
            << std::setw(14) << "LS "
            << std::setw(14) << "norm(A)"
            << std::setw(14) << "cond(A)"
            << std::endl;

  std::cout << std::setprecision(7)
            << std::setw(6)  << iter
            << std::setw(14) << x1
            << std::setw(14) << xnorm
            << std::setw(14) << rnorm
            << std::setw(14) << Arnorm
            << std::setw(14) << relres
            << std::setw(14) << relAres
            << std::setw(14) << Anorm
            << std::setw(14) << Acond
            << "\n\n"
            << std::flush;
}

template<typename DerivedOP, typename FloatType>
void RealSolver<DerivedOP, FloatType>::solve(BaseInterface<DerivedOP, REAL<FloatType> >  & client) const
{
  const int n = client.n;
  const std::vector<FloatType> &b = client.b, zeros(n,0);
  std::vector<FloatType>& x = client.x;
  // local constants
  const FloatType EPSINV = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10))),
		              NORMMAX = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10)/2.));
  // local arrays and variables
  const FloatType shift_ = client.shift;
  const bool checkA_ = true, precon_ = client.useMsolve, disable_ = client.disable;
  FloatType rtol_ = client.rtol, maxxnorm_ = std::min({client.maxxnorm, 1./eps_}),
            trancond_ = std::min({client.trancond, NORMMAX}),
            Acondlim_ = std::min({client.Acondlim, EPSINV});
  FloatType Arnorm_ = 0, xnorm_ = 0, Anorm_ = 0, Acond_ = 1;
  int itnlim_ = client.itnlim, istop_ = 0, itn_ = 0;
  std::vector<FloatType> r1(n), r2(n), v(n), w(n), wl(n), wl2(n), xl2(n), y(n), vec2(2), vec3(3);
  FloatType Axnorm = 0, beta = 0, beta1 = dnrm2_(n, &b[0], 1),
            betan = 0, ieps = 0.1/eps_, pnorm = 0, relAres = 0, relAresl = 0, relresl = 0,
            t1 = 0, t2 = 0, xl2norm = 0, cr1 = -1, cr2 = -1, cs = -1,
            dbar = 0, dltan = 0, epln = 0, eplnn = 0, eta = 0, etal = 0,
            etal2 = 0, gama = 0, gama_QLP = 0, gamal = 0, gamal_QLP = 0, gamal_tmp = 0,
            gamal2 = 0, gamal3 = 0, gmin = 0, gminl = 0, phi = 0, s = 0,
            sn = 0, sr1 = 0, sr2 = 0, t = 0, tau = 0, taul = 0, taul2 = 0, u = 0,
            u_QLP = 0, ul = 0, ul_QLP = 0, ul2 = 0, ul3 = 0, ul4 = 0, vepln = 0,
            vepln_QLP = 0, veplnl = 0, veplnl2 = 0, x1last = 0, xnorml = 0, Arnorml = 0,
            Anorml = 0, rnorml = 0, Acondl = 0;
  int QLPiter = 0, flag0 = 0;
  bool done = false, lastiter = false, likeLS;
  const std::vector<std::string> msg = {
         "beta_{k+1} < eps.                                                ", //  1
         "beta2 = 0.  If M = I, b and x are eigenvectors of A.             ", //  2
         "beta1 = 0.  The exact solution is  x = 0.                        ", //  3
         "A solution to (poss. singular) Ax = b found, given rtol.         ", //  4
         "A solution to (poss. singular) Ax = b found, given eps.          ", //  5
         "Pseudoinverse solution for singular LS problem, given rtol.      ", //  6
         "Pseudoinverse solution for singular LS problem, given eps.       ", //  7
         "The iteration limit was reached.                                 ", //  8
         "The operator defined by Aprod appears to be unsymmetric.         ", //  9
         "The operator defined by Msolve appears to be unsymmetric.        ", //  10
         "The operator defined by Msolve appears to be indefinite.         ", //  11
         "xnorm has exceeded maxxnorm or will exceed it next iteration.    ", //  12
         "Acond has exceeded Acondlim or 0.1/eps.                          ", //  13
         "Least-squares problem but no converged solution yet.             ", //  14
         "A null vector obtained, given rtol.                              "};//  15
  x = zeros, xl2 = zeros;
  if (client.print)
  {
    std::cout << std::setprecision(3);
    std::cout << std::endl
              << std::setw(54) << "Enter MINRES-QLP(INFO)" << std::endl
              << "  "
              << "\n\n"
              << std::setw(14) << "n  = "        << std::setw(8) << n
              << std::setw(14) << "||b||  = "    << std::setw(8) << beta1
              << std::setw(14) << "precon  = "   << std::setw(8) << ((precon_) ? "true" : "false")
              << std::endl
              << std::setw(14) << "itnlim  = "   << std::setw(8) << itnlim_
              << std::setw(14) << "rtol  = "     << std::setw(8) << rtol_
              << std::setw(14) << "shift  = "    << std::setw(8) << shift_
              << std::endl
              << std::setw(14) << "raxxnorm  = " << std::setw(8) << maxxnorm_
              << std::setw(14) << "Acondlim  = " << std::setw(8) << Acondlim_
              << std::setw(14) << "trancond  = " << std::setw(8) << trancond_
              << std::endl
              << "  "
              << std::endl << std::endl
              << std::flush;
  }
  y = b, r1 = b;
  if (precon_)
    client.Msolve(n, &b[0], &y[0]);
  beta1 = std::inner_product(b.begin(), b.end(), y.begin(), 0.0);
  if (beta1 < 0 && dnrm2_(n, &y[0], 1) > eps_)
    istop_ = 11;
  if (beta1 == 0)
    istop_ = 3;
  beta1 = std::sqrt(beta1);

  if (checkA_ && precon_)
  {
    client.Msolve(n, &y[0], &r2[0]);
    s = std::inner_product(y.begin(), y.end(), y.begin(), 0.0),
    t = std::inner_product(r1.begin(), r1.end(), r2.begin(), 0.0);
    FloatType z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
    if (z > epsa)
      istop_ = 10;
  }

  if (checkA_)
  {
    client.Aprod(n, &y[0], &w[0]);
    client.Aprod(n, &w[0], &r2[0]);
    s = std::inner_product(w.begin(), w.end(), w.begin(), 0.0),
    t = std::inner_product(y.begin(), y.end(), r2.begin(), 0.0);
    FloatType z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
    if (z > epsa)
      istop_ = 9;
  }
  betan = beta1, phi = beta1;
  FloatType rnorm_ = betan;
  FloatType relres = rnorm_ / (Anorm_*xnorm_ + beta1);
  r2 = b, w = zeros, wl = zeros, done = false;
  // MINRESQLP iteration loop.
  while(istop_ <= flag0)
  {
    itn_ += 1;
    FloatType betal = beta;
    beta = betan;
    s = 1./beta;
    for(int index=0; index<n; ++index)
      v[index] = s*y[index];
    client.Aprod(n, &v[0], &y[0]);
    if (shift_ != 0)
      for(int index=0; index<n; ++index)
        y[index] -= shift_*v[index];
    if (itn_ >= 2)
      for(int index=0; index<n; ++index)
        y[index] += (-beta/betal)*r1[index];
    FloatType alfa = std::inner_product(v.begin(), v.end(), y.begin(), 0.0);
    for(int index=0; index<n; index++)
      y[index] = y[index] + (-alfa/beta)*r2[index];
    r1 = r2, r2 = y;
    if (!precon_)
      betan = dnrm2_(n, &y[0], 1);
    else
    {
      client.Msolve(n, &r2[0], &y[0]);
      betan = std::inner_product(r2.begin(), r2.end(), y.begin(), 0.0);
      if (betan > 0)
        betan = std::sqrt(betan);
      else if (dnrm2_(n, &y[0], 1) > eps_)
      {
        istop_ = 11;
        break;
      }
    }

    if (itn_ == 1)
    {
      vec2[0] = alfa, vec2[1] = betan;
      pnorm = dnrm2_(2, &vec2[0], 1);
    }
    else
    {
      vec3[0] = beta, vec3[1] = alfa, vec3[2] = betan;
      pnorm = dnrm2_(3, &vec3[0], 1);
    }
    dbar = dltan;
    FloatType dlta = cs*dbar + sn*alfa;
    epln = eplnn;
    FloatType gbar = sn*dbar - cs*alfa;
    eplnn = sn*betan, dltan = -cs*betan;
    FloatType dlta_QLP = dlta;
    gamal3 = gamal2, gamal2 = gamal, gamal  = gama;
    symortho_(gbar, betan, cs, sn, gama);
    FloatType gama_tmp = gama;
    taul2 = taul;
    taul = tau;
    tau = cs*phi;
    phi = sn*phi;
    Axnorm = std::sqrt(std::pow(Axnorm,2) + std::pow(tau,2));
    // apply the previous right reflection P{k-2,k}
    if (itn_ > 2)
    {
      veplnl2 = veplnl;
      etal2 = etal;
      etal = eta;
      FloatType dlta_tmp = sr2*vepln - cr2*dlta;
      veplnl = cr2 * vepln + sr2 * dlta;
      dlta = dlta_tmp;
      eta = sr2 * gama;
      gama = -cr2 * gama;
    }

    // compute the current right reflection P{k-1,k}, P_12, P_23,...
    if (itn_ > 1)
    {
      symortho_(gamal, dlta, cr1, sr1, gamal_tmp);
      gamal = gamal_tmp;
      vepln = sr1 * gama;
      gama = -cr1 * gama;
    }

    // update xnorm
    FloatType xnorml = xnorm_;
    ul4 = ul3;
    ul3 = ul2;

    if (itn_ > 2)
      ul2 = ( taul2 - etal2*ul4 - veplnl2*ul3 ) / gamal2;
    if (itn_ > 1)
      ul = ( taul - etal*ul3 - veplnl*ul2) / gamal;

    vec3[0] = xl2norm, vec3[1] = ul2, vec3[2] = ul;
    FloatType xnorm_tmp = dnrm2_(3, &vec3[0], 1);  // norm([xl2norm ul2 ul]);

    if (std::abs(gama) > eps_)
    {
      u = (tau - eta*ul2 - vepln*ul) / gama;
      likeLS = relAresl < relresl;
      vec2[0] = xnorm_tmp, vec2[1] = u;
      if (likeLS && dnrm2_(2, &vec2[0], 1) > maxxnorm_)
      {
        u = 0;
        istop_ = 12;
      }
    }
    else
    {
      u = 0;
      istop_ = 14;
    }

    vec2[0] = xl2norm, vec2[1] = ul2;
    xl2norm = dnrm2_(2, &vec2[0], 1);
    vec3[0] = xl2norm, vec3[1] = ul, vec3[2] = u;
    xnorm_ = dnrm2_(3, &vec3[0], 1);

    // MINRES updates
    if (Acond_ < trancond_ && istop_ == flag0 && QLPiter == 0)
    {
      wl2 = wl;
      wl = w;
      if (gama_tmp > eps_)
      {
        s = 1./gama_tmp;
        for (int index=0; index<n; ++index)
          w[index] = (v[index] - epln*wl2[index] - dlta_QLP*wl[index])*s;
      }
      if (xnorm_ < maxxnorm_)
      {
        x1last = x[0];
        for (int index=0; index<n; ++index)
          x[index] += tau*w[index];
      }
      else
        istop_ = 12, lastiter = true;
    }
    else
    {
      QLPiter += 1;
      if (QLPiter == 1)
      {
        xl2 = zeros;
        if (itn_ > 1) // construct w_{k-3}, w_{k-2}, w_{k-1}
        {
          if (itn_ > 3)
            for (int index=0; index<n; ++index)
              wl2[index] = gamal3*wl2[index] + veplnl2*wl[index] + etal*w[index];
          if (itn_ > 2)
            for (int index=0; index<n; ++index)
              wl[index] = gamal_QLP*wl[index] + vepln_QLP*w[index];
          for (int index=0; index<n; ++index)
            w[index] *= gama_QLP;
          for (int index=0; index<n; ++index)
            xl2[index] = x[index] - ul_QLP*wl[index] - u_QLP*w[index];
        }
      }
      if (itn_ == 1)
      {
        wl2 =  wl;
        for (int index=0; index<n; ++index)
          wl[index] = sr1*v[index];
        for (int index=0; index<n; ++index)
          w[index] = -cr1*v[index];
      }
      else if (itn_ == 2)
      {
        wl2 = wl;
        for (int index=0; index<n; ++index)
          wl[index] = cr1*w[index] + sr1*v[index];
        for (int index=0; index<n; ++index)
          w[index] = sr1*w[index] - cr1*v[index];
      }
      else
      {
        wl2 = wl;
        wl = w;
        for (int index=0; index<n; ++index)
          w[index] = sr2*wl2[index] - cr2*v[index];
        for (int index=0; index<n; ++index)
          wl2[index] = cr2*wl2[index] + sr2*v[index];
        for (int index=0; index<n; ++index)
          v[index] = cr1*wl[index] + sr1*w[index];
        for (int index=0; index<n; ++index)
          w[index] = sr1*wl[index] - cr1*w[index];
        wl = v;
	  }
      x1last = x[0];
      for (int index=0; index<n; ++index)
        xl2[index] = xl2[index] + ul2*wl2[index];
      for (int index=0; index<n; ++index)
        x[index] = xl2[index] + ul*wl[index] + u*w[index];
    }
    // compute the next right reflection P{k-1,k+1}
    gamal_tmp = gamal;
    symortho_(gamal_tmp, eplnn, cr2, sr2, gamal);
    // store quantities for transfering from MINRES to MINRESQLP
    gamal_QLP = gamal_tmp;
    vepln_QLP = vepln;
    gama_QLP = gama;
    ul_QLP = ul;
    u_QLP = u;
    // estimate various norms
    FloatType abs_gama = abs(gama);
    Anorml = Anorm_;
    Anorm_ = std::max({Anorm_, pnorm, gamal, abs_gama});
    if (itn_ == 1)
    {
      gmin  = gama;
      gminl = gmin;
    }
    else if (itn_ > 1)
    {
      FloatType gminl2 = gminl;
      gminl = gmin;
      vec3[0] = gminl2, vec3[1] = gamal, vec3[2] = abs_gama;
      gmin = std::min({gminl2, gamal, abs_gama});
    }
    FloatType Acondl = Acond_;
    Acond_ = Anorm_ / gmin;
    FloatType rnorml   = rnorm_;
    relresl = relres;
    if (istop_ != 14)
      rnorm_ = phi;
    relres = rnorm_ / (Anorm_ * xnorm_ + beta1);
    vec2[0] = gbar, vec2[1] = dltan;
    FloatType rootl = dnrm2_(2, &vec2[0], 1);
    Arnorml  = rnorml * rootl;
    relAresl = rootl / Anorm_;
    // see if any of the stopping criteria are satisfied.
    FloatType epsx = Anorm_*xnorm_*eps_;
    if (istop_ == flag0 || istop_ == 14)
      t1 = 1. + relres, t2 = 1. + relAresl;
    if (t1 <= 1)
      istop_ = 5;
    else if (t2 <= 1)
      istop_ = 7;
    else if (relres <= rtol_)
      istop_ = 4;
    else if (relAresl <= rtol_)
      istop_ = 6;
    else if (epsx >= beta1)
      istop_ = 2;
    else if (xnorm_ >= maxxnorm_)
      istop_ = 12;
    else if (Acond_ >= Acondlim_ || Acond_ >= ieps)
      istop_ = 13;
    else if (itn_ >= itnlim_)
      istop_ = 8;
    else if (betan < eps_)
      istop_ = 1;
    if (disable_ && itn_ < itnlim_)
    {
      istop_ = flag0, done = false;
      if (Axnorm < rtol_*Anorm_*xnorm_)
        istop_ = 15, lastiter = false;
    }
	  if (istop_ != flag0)
	  {
      done = true;
      if (istop_ == 6 || istop_ == 7 || istop_ == 12 || istop_ == 13)
        lastiter = true;
      if (lastiter)
        itn_ -= 1, Acond_ = Acondl, rnorm_ = rnorml, relres = relresl;
      client.Aprod(n, &x[0], &r1[0]);
      for (int index=0; index<n; ++index)
        r1[index] = b[index] - r1[index] + shift_*x[index];
      client.Aprod(n, &r1[0], &wl2[0]);
      for (int index=0; index<n; ++index)
        wl2[index] = wl2[index] - shift_*r1[index];
      Arnorm_ = dnrm2_(n, &wl2[0], 1);
      if (rnorm_ > 0 && Anorm_ > 0)
        relAres = Arnorm_ / (Anorm_*rnorm_);
    }
    if(client.print)
      printstate_(itn_-1, x1last, xnorml, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl);
  }
  // end of iteration loop.
  client.istop = istop_, client.itn = itn_, client.rnorm = rnorm_;
  client.Arnorm = Arnorm_, client.xnorm = xnorm_, client.Anorm = Anorm_;
  client.Acond  = Acond_;
  if (client.print)
  {
    printstate_(itn_, x[0], xnorm_, rnorm_, Arnorm_, relres, relAres, Anorm_, Acond_);
    std::cout << "  " << "Exit MINRES-QLP" << ": "
              << msg[istop_-1] << "\n\n";
  }
}


template<typename DerivedOP, typename FloatType>
void HermitianSolver<DerivedOP, FloatType>::printstate_(const int iter,
  const std::complex<FloatType> x1, const FloatType xnorm,
  const FloatType rnorm, const FloatType Arnorm, const FloatType relres,
  const FloatType relAres, const FloatType Anorm,  const FloatType Acond) const
{
  std::cout << std::setw(7) << "iter "
            << std::setw(21) << "x[0] "
            << std::setw(14) << "xnorm "
            << std::setw(14) << "rnorm "
            << std::setw(14) << "Arnorm "
            << std::setw(14) << "Compatible "
            << std::setw(14) << "LS "
            << std::setw(14) << "norm(A)"
            << std::setw(14) << "cond(A)"
            << std::endl;
  std::cout << std::setprecision(4)
            << std::setw(6)  << iter
            << std::setw(21) << x1
            << std::setw(14) << xnorm
            << std::setw(14) << rnorm
            << std::setw(14) << Arnorm
            << std::setw(14) << relres
            << std::setw(14) << relAres
            << std::setw(14) << Anorm
            << std::setw(14) << Acond
            << "\n\n"
            << std::flush;
}

template<typename DerivedOP, typename FloatType>
std::complex<FloatType> HermitianSolver<DerivedOP, FloatType>::zdotc_(const int n, const std::complex<FloatType>* cx,
  const int incx, const std::complex<FloatType>* cy, const int incy) const
{
  std::complex<FloatType> ctemp = std::complex<FloatType>(0,0);
  int ix, iy;
  if (n <= 0)
    return std::complex<FloatType>(0,0);
  if (incx == 1 && incy == 1)
    for(int i = 0; i<n; ++i)
      ctemp += std::conj(cx[i])*cy[i];
  else
  {
    if (incx >= 0)
      ix = 1;
    else
      ix = (-n + 1)*incx + 1;
    if (incy >= 0)
      iy = 1;
    else
      iy = (-n + 1)*incy + 1;
    for(int i=0; i<n; ++i)
    {
      ctemp += std::conj(cx[ix])*cy[iy];
      ix += incx;
      iy += incy;
    }
  }
  return ctemp;
}

template<typename DerivedOP, typename FloatType>
FloatType HermitianSolver<DerivedOP, FloatType>::znrm2_(const int n, const std::complex<FloatType>* x, const int incx) const
{
  FloatType norm, scale, ssq;
  if (n< 1 || incx < 1)
    norm = 0;
  else
  {
    scale = 0, ssq = 1;
    for (int ix = 0; ix<1+(n - 1)*incx; ix += incx)
    {
      if (x[ix].real() != 0)
      {
        FloatType temp = std::abs(x[ix].real());
        if (scale < temp)
        {
          ssq = 1 + ssq*std::pow(scale/temp, 2);
          scale = temp;
        }
        else
          ssq += std::pow(temp/scale, 2);
      }
      if (x[ix].imag()!= 0)
      {
        FloatType temp = std::abs(x[ix].imag());
        if (scale < temp)
        {
          ssq = 1 + ssq*std::pow(scale/temp, 2);
          scale = temp;
        }
        else
          ssq += std::pow(temp / scale, 2);
      }
    }
    norm  = scale*std::sqrt (ssq);
  }
  return norm;
}

template<typename DerivedOP, typename FloatType>
void HermitianSolver<DerivedOP, FloatType>::zsymortho_(const std::complex<FloatType>& a, const std::complex<FloatType>& b,
  FloatType& c, std::complex<FloatType>& s, std::complex<FloatType>& r) const
{
  FloatType t, abs_a = std::abs(a), abs_b = std::abs(b);
  if (abs_b <= realmin_)
    c = 1., s = std::complex<FloatType>(0,0), r = a;
  else if (abs_a <= realmin_)
    c = 0, s = 1, r = b;
  else if (abs_b > abs_a)
  {
    t = abs_a/abs_b;
    c = 1./std::sqrt(1. + std::pow(t,2));
    s = c*std::conj((b/abs_b) / (a/abs_a));
    c = c*t;
    r = b/std::conj(s);
  }
  else
  {
    t = abs_b/abs_a;
    c = 1. / std::sqrt(1. + std::pow(t,2));
    s = c*t*std::conj((b/abs_b) / (a/abs_a));
    r = a/c;
  }
}

template<typename DerivedOP, typename FloatType>
void HermitianSolver<DerivedOP, FloatType>::solve(BaseInterface<DerivedOP, IMAG<FloatType> > & client) const
{
  const int n = client.n;
  const std::vector<std::complex<FloatType> > &b = client.b, zeros(n,0);
  const std::complex<FloatType> zero = std::complex<FloatType>(0,0);
  std::vector<std::complex<FloatType> >& x = client.x;
  // local arrays and variables
  FloatType shift_, rtol_, maxxnorm_, trancond_, Acondlim_,
            rnorm_, Arnorm_, xnorm_ = 0, Anorm_ = 0, Acond_ = 1.;
  bool checkA_, precon_, disable_;
  int itnlim_, nout_, istop_, itn_ = 0;
  std::vector<std::complex<FloatType> > r1(n), r2(n), v(n), w(n), wl(n), wl2(n), xl2(n), y(n), vec2(2), vec3(3);
  FloatType  Axnorm = 0, beta = 0, gmin = 0, gminl = 0, pnorm = 0, relAres = 0,
             relAresl = 0, relresl = 0, t1 = 0, t2 = 0, xl2norm = 0, cr1 = -1,
             cr2 = -1, cs = -1;
  std::complex<FloatType> dbar = zero, dltan = zero, eplnn = zero, eta = zero,
                          etal = zero, etal2 = zero, gama = zero, gama_QLP = zero,
                          gamal = zero, gamal_QLP = zero, gamal_tmp = zero,
                          gamal2 = zero, s = zero, sn = zero, sr1 = zero,
                          sr2 = zero, t = zero, tau = zero, taul = zero, u = zero,
                          u_QLP = zero, ul = zero, ul_QLP = zero, ul2 = zero,
                          ul3 = zero, vepln = zero, vepln_QLP = zero,
                          veplnl = zero, veplnl2 = zero, x1last = x[0];
  int QLPiter = 0, flag0 = 0;
  bool done, lastiter, likeLS;
  // local constants
  const FloatType EPSINV  = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10))),
                  NORMMAX = std::pow(10.0, std::floor(std::log(1./eps_)/std::log(10)/2.));
  const std::vector<std::string> msg = {
         "beta_{k+1} < eps.                                                ", // 1
         "beta2 = 0.  If M = I, b and x are eigenvectors of A.             ", // 2
         "beta1 = 0.  The exact solution is  x = 0.                        ", // 3
         "A solution to (poss. singular) Ax = b found, given rtol.         ", // 4
         "A solution to (poss. singular) Ax = b found, given eps.          ", // 5
         "Pseudoinverse solution for singular LS problem, given rtol.      ", // 6
         "Pseudoinverse solution for singular LS problem, given eps.       ", // 7
         "The iteration limit was reached.                                 ", // 8
         "The operator defined by Aprod appears to be non-Hermitian.       ", // 9
         "The operator defined by Msolve appears to be non-Hermitian.      ", // 10
         "The operator defined by Msolve appears to be indefinite.         ", // 11
         "xnorm has exceeded maxxnorm  or will exceed it next iteration.   ", // 12
         "Acond has exceeded Acondlim or 0.1/eps.                          ", // 13
         "Least-squares problem but no converged solution yet.             ", // 14
         "A null vector obtained, given rtol.                              "};// 15
  shift_ = client.shift,
  checkA_ = true;
  disable_ = client.disable;
  itnlim_ = client.itnlim;
  rtol_ = client.rtol;
  maxxnorm_ = std::min({client.maxxnorm, static_cast<FloatType>(1.0)/eps_});
  trancond_ = std::min({client.trancond, NORMMAX});
  Acondlim_ = client.Acondlim;
  precon_ = client.useMsolve;
  lastiter = false;
  istop_ = flag0;
  FloatType beta1 = znrm2_(n, &b[0], 1), ieps = 0.1/eps_;
  x = zeros;
  xl2 = zeros;
  x1last = x[0];
  y  = b;
  r1 = b;

  if (precon_)
    client.Msolve(n, &b[0], &y[0]);
  beta1 = (zdotc_(n, &b[0], 1, &y[0], 1)).real();

  if (beta1<0 && znrm2_(n, &y[0], 1)>eps_)
    istop_ = 11;
  if (beta1 == 0)
    istop_ = 3;
  beta1 = std::sqrt(beta1);
  if (checkA_ && precon_)
  {
    client.Msolve(n, &y[0], &r2[0]);
    s = zdotc_(n, &y[0], 1, &y[0], 1);
    t = zdotc_(n, &r1[0], 1, &r2[0], 1);
    FloatType z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
    if (z > epsa)
      istop_ = 10;
  }

  if (checkA_)
  {
    client.Aprod(n, &y[0], &w[0]);
    client.Aprod(n, &w[0], &r2[0]);
    s = zdotc_(n, &w[0], 1, &w[0], 1);
    t = zdotc_(n, &y[0], 1, &r2[0], 1);
    FloatType z = std::abs(s-t), epsa = (std::abs(s) + eps_)*std::pow(eps_, 0.33333);
    if (z > epsa)
	  istop_ = 9;
  }

  FloatType betan = beta1;
  std::complex<FloatType> phi = std::complex<FloatType>(beta1, 0);
  rnorm_ = betan;
  FloatType relres = rnorm_ / (Anorm_*xnorm_ + beta1);
  relAres= 0;
  r2 = b;
  w = zeros;
  wl = zeros;
  done = false;

  if (client.print)
  {
    std::cout << std::setprecision(3);
    std::cout << std::endl
              << std::setw(54) << "Enter MINRES-QLP(INFO)" << std::endl
              << "  "
              << "\n\n"
              << std::setw(14) << "n  = "        << std::setw(8) << n
              << std::setw(14) << "||b||  = "    << std::setw(8) << beta1
              << std::setw(14) << "precon  = "   << std::setw(8) << ((precon_) ? "true" : "false")
              << std::endl
              << std::setw(14) << "itnlim  = "   << std::setw(8) << itnlim_
              << std::setw(14) << "rtol  = "     << std::setw(8) << rtol_
              << std::setw(14) << "shift  = "    << std::setw(8) << shift_
              << std::endl
              << std::setw(14) << "raxxnorm  = " << std::setw(8) << maxxnorm_
              << std::setw(14) << "Acondlim  = " << std::setw(8) << Acondlim_
              << std::setw(14) << "trancond  = " << std::setw(8) << trancond_
              << std::endl
              << "  "
              << std::endl << std::endl
              << std::flush;
  }

  // main iteration loop.
  while (istop_ <= flag0)
  {
    itn_ += 1; // k = itn = 1 first time through
    FloatType betal = beta;               // betal = betak
    beta = betan;
    s = 1./beta;         // Normalize previous vector (in y).
    for (int index=0; index<n; ++index)
      v[index] = s*y[index]; // v = vk if P = I.
    client.Aprod(n, &v[0], &y[0]);
    if (std::abs(shift_) >= realmin_)
      for (int index=0; index<n; ++index)
        y[index] += -shift_*v[index];
    if (itn_ >= 2)
      for (int index=0; index<n; ++index)
		y[index] += (-beta/betal)*r1[index];
    FloatType alfa = (zdotc_(n, &v[0], 1, &y[0], 1)).real();
    for (int index=0; index<n; ++index)
      y[index] += (-alfa/beta)*r2[index];
    r1 = r2;
    r2 = y;
    if (!precon_)
      betan = znrm2_(n, &y[0], 1);
    else
    {
      client.Msolve(n, &r2[0], &y[0]);
      betan = (zdotc_(n, &r2[0], 1, &y[0], 1)).real();
      if (betan > 0)
        betan = std::sqrt(betan);
      else if (znrm2_(n, &y[0], 1) > eps_)
      { // M must be indefinite.
        istop_ = 11;
        break;
      }
    }
    if (itn_ == 1)
    {
      vec2[0] = alfa, vec2[1] = betan;
      pnorm = znrm2_(2, &vec2[0], 1);
    }
    else
    {
      vec3[0] = beta, vec3[1] = alfa, vec3[2] = betan;
      pnorm = znrm2_(3, &vec3[0], 1);
    }

    // Apply previous left reflection Qk-1 to get
    //   [deltak epslnk+1] = [cs  sn][dbark    0   ]
    //   [gbar k dbar k+1]   [sn -cs][alfak betak+1].
    dbar = dltan;
    std::complex<FloatType> dlta = cs*dbar + sn*alfa; // dlta1  = 0         deltak
    std::complex<FloatType> epln = eplnn;
    std::complex<FloatType> gbar = sn*dbar - cs*alfa; // gbar 1 = alfa1     gbar k
                            eplnn = sn*betan; // eplnn2 = 0         epslnk+1
                            dltan = -cs*betan; // dbar 2 = beta2     dbar k+1
    std::complex<FloatType> dlta_QLP = dlta;
    // Compute the current left reflection Qk
    std::complex<FloatType> gamal3 = gamal2;
    gamal2 = gamal;
    gamal = gama;
    zsymortho_(gbar, std::complex<FloatType>(betan,0), cs, sn, gama);
    std::complex<FloatType> gama_tmp = gama;
    std::complex<FloatType> taul2  = taul;
    taul = tau;
    tau = cs * phi;
    phi = sn * phi;                   //  phik
    Axnorm = std::sqrt( std::pow(Axnorm,2) + std::pow(std::abs(tau), 2));
    // Apply the previous right reflection P{k-2,k}
    if (itn_ > 2)
    {
      veplnl2 = veplnl;
      etal2 = etal;
      etal = eta;
      std::complex<FloatType> dlta_tmp = sr2 * vepln - cr2 * dlta;
      veplnl = cr2 * vepln + sr2 * dlta;
      dlta = dlta_tmp;
      eta = sr2 * gama;
      gama = -cr2 * gama;
    }
    // Compute the current right reflection P{k-1,k}, P_12, P_23,...
    if (itn_ > 1)
    {
      zsymortho_(gamal, dlta, cr1, sr1, gamal_tmp);
      gamal = gamal_tmp;
      vepln = sr1 * gama;
      gama = -cr1 * gama;
    }
    // Update xnorm
    FloatType xnorml = xnorm_;
    std::complex<FloatType> ul4 = ul3;
    ul3 = ul2;
    if (itn_ > 2)
      ul2 = (taul2 - etal2 * ul4 - veplnl2 * ul3) / gamal2;
    if (itn_ > 1)
      ul = (taul - etal * ul3 - veplnl * ul2) / gamal;
    vec3[0] = xl2norm, vec3[1] = ul2, vec3[2] = ul;
    FloatType xnorm_tmp = znrm2_(3, &vec3[0], 1);  // norm([xl2norm ul2 ul]);
    if (std::abs(gama) > 0  &&  xnorm_tmp < maxxnorm_)
    {
      u = (tau - eta*ul2 - vepln*ul) / gama;
      likeLS = relAresl < relresl;
      vec2[0] = xnorm_tmp;
      vec2[1] = u;
      if (likeLS && znrm2_(2, &vec2[0], 1) > maxxnorm_)
      {
        u = 0;
        istop_ = 12;
      }
    }
    else
    {
      u = 0;
      istop_ = 14;
    }
    vec2[0] = xl2norm, vec2[1] = ul2;
    xl2norm = znrm2_(2, &vec2[0], 1);
    vec3[0] = xl2norm, vec3[1] = ul, vec3[2] = u;
    xnorm_  = znrm2_(3, &vec3[0], 1);
    if (Acond_ < trancond_ && istop_ == flag0 && QLPiter == 0) // MINRES updates
    {
      wl2 = wl;
      wl = w;
      if (std::abs(gama_tmp) > 0)
      {
        s = static_cast<FloatType>(1.0) / gama_tmp;
        for(int index=0; index<n; ++index)
          w[index] = (v[index] - epln*wl2[index] - dlta_QLP*wl[index])*s;
      }
      if (xnorm_ < maxxnorm_)
      {
        x1last = x[0];
        for(int index=0; index<n; ++index)
          x[index] = x[index] + tau*w[index];
      }
      else
      {
        istop_ = 12;
        lastiter = true;
      }
    }
    else //MINRES-QLP updates
    {
      QLPiter += 1;
      if (QLPiter == 1)
      {
        xl2 = zeros; // vector
        if (itn_ > 1)
        {
          // construct w_{k-3}, w_{k-2}, w_{k-1}
          if (itn_ > 3)
            for (int index=0; index<n; ++index)
              wl2[index] = gamal3*wl2[index] + veplnl2*wl[index] + etal*w[index];
          // w_{k-3}
          if (itn_ > 2)
            for (int index=0; index<n; ++index)
              wl[index] = gamal_QLP*wl[index] + vepln_QLP*w[index];
          // w_{k-2}
          for (int index=0; index<n; ++index)
			      w[index] = gama_QLP*w[index];
          for (int index=0; index<n; ++index)
            xl2[index] =  x[index] - ul_QLP*wl[index] - u_QLP*w[index];
        }
      }
      if (itn_ == 1)
      {
        wl2 =  wl;
        for (int index=0; index<n; ++index)
          wl[index]  =  sr1*v[index];
        for (int index=0; index<n; ++index)
          w[index]   = -cr1*v[index];
      }
      else if (itn_ == 2)
      {
        wl2 = wl;
        for (int index=0; index<n; ++index)
          wl[index]  = cr1*w[index] + sr1*v[index];
        for (int index=0; index<n; ++index)
          w[index] = sr1*w[index] - cr1*v[index];
      }
      else
      {
        wl2 = wl;
        wl  = w;
        for (int index=0; index<n; ++index)
          w[index] = sr2*wl2[index] - cr2*v[index];
        for (int index=0; index<n; ++index)
          wl2[index] = cr2*wl2[index] + sr2*v[index];
        for (int index=0; index<n; ++index)
          v[index] = cr1*wl[index] + sr1*w[index];
        for (int index=0; index<n; ++index)
          w[index] = sr1*wl[index] - cr1*w[index];
        wl  = v;
      }
      x1last = x[0];
      for (int index=0; index<n; ++index)
        xl2[index] += ul2*wl2[index];
      for (int index=0; index<n; ++index)
        x[index] = xl2[index] + ul*wl[index] + u*w[index];
    }
    // Compute the next right reflection P{k-1,k+1}
    gamal_tmp = gamal;
    zsymortho_(gamal_tmp, eplnn, cr2, sr2, gamal);
    // Store quantities for transfering from MINRES to MINRES-QLP
    gamal_QLP = gamal_tmp;
    vepln_QLP = vepln;
    gama_QLP = gama;
    ul_QLP = ul;
    u_QLP = u;
    // Estimate various norms
    FloatType abs_gama = std::abs(gama);
    FloatType Anorml = Anorm_;
    Anorm_ = std::max({Anorm_, pnorm, abs(gamal), abs_gama});
    if (itn_ == 1)
    {
      gmin  = abs_gama;
      gminl = gmin;
    }
    else if (itn_ > 1)
    {
      FloatType gminl2 = gminl;
      gminl = gmin;
      vec3[0] = gminl2, vec3[1] = gamal, vec3[2] = abs_gama;
      gmin = std::min({gminl2, std::abs(gamal), abs_gama});
    }
    FloatType Acondl = Acond_;
    Acond_ = Anorm_ / gmin;
    FloatType rnorml = rnorm_;
    relresl = relres;
    if (istop_ != 14)
      rnorm_ = std::abs(phi);
    relres = rnorm_ / (Anorm_ * xnorm_ + beta1);
    vec2[0] = gbar;
    vec2[1] = dltan;
    FloatType rootl = znrm2_(2, &vec2[0], 1);
    FloatType Arnorml = rnorml * rootl;
    relAresl = rootl / Anorm_;
    // See if any of the stopping criteria are satisfied.
    FloatType epsx = Anorm_*xnorm_*eps_;
    if (istop_ == flag0 || istop_ == 14)
    {
      t1 = 1.0 + relres;
      t2 = 1.0 + relAresl;
    }
    if (t1 <= 1.)
      istop_ = 5;                           // Accurate Ax=b solution
    else if (t2 <= 1)
      istop_ = 7;                           // Accurate LS solution
    else if (relres   <= rtol_)
      istop_ = 4;                           // Good enough Ax=b solution
    else if (relAresl <= rtol_)
      istop_ = 6;                           // Good enough LS solution
    else if (epsx >= beta1)
      istop_ = 2;                           // x is an eigenvector
    else if (xnorm_ >= maxxnorm_)
      istop_ = 12;                          // xnorm exceeded its limit
    else if (Acond_ >= Acondlim_ || Acond_ >= ieps)
      istop_ = 13;                          // Huge Acond
    else if (itn_ >= itnlim_)
      istop_ = 8;                           // Too many itns
    else if (betan < eps_)
      istop_ = 1;                           // Last iteration of Lanczos

    if (disable_ && itn_ < itnlim_)
    {
      istop_ = flag0;
      done = false;
      if (Axnorm < rtol_*Anorm_*xnorm_)
      {
        istop_ = 15;
        lastiter = false;
      }
    }
    if (istop_ != flag0)
    {
      done = true;
      if (istop_ == 6 || istop_ == 7 || istop_ == 12 || istop_ == 13)
        lastiter = true;
      if (lastiter)
      {
        itn_ -= 1;
        Acond_ = Acondl;
        rnorm_ = rnorml;
        relres = relresl;
      }
      client.Aprod(n, &x[0], &r1[0]);
      for (int index=0; index<n; ++index)
        r1[index]  = b[index] - r1[index] + shift_*x[index]; // r1 to temporarily store residual vector
      client.Aprod(n, &r1[0], &wl2[0]); // wl2 to temporarily store A*r1
      for (int index=0; index<n; ++index)
        wl2[index] = wl2[index] - shift_*r1[index];
      Arnorm_ = znrm2_(n, &wl2[0], 1);
      if (rnorm_ > 0 && Anorm_ > 0)
        relAres = Arnorm_ / (Anorm_*rnorm_);
    }
    if (client.print)
    {
      if (itn_ <= 11 || itn_%10 == 1)
      {
        printstate_(itn_-1, x1last, xnorml, rnorml, Arnorml, relresl, relAresl, Anorml, Acondl);
        if (itn_ == 11)
		      std::cout << std::endl;
      }
    }
    if (istop_ != flag0)
      break;
  } // end of iteration loop.
  client.istop = istop_, client.itn = itn_, client.rnorm = rnorm_;
  client.Arnorm = Arnorm_, client.xnorm = xnorm_, client.Anorm = Anorm_;
  client.Acond = Acond_;
  if (client.print)
  {
    printstate_(itn_, x[0], xnorm_, rnorm_, Arnorm_, relres, relAres, Anorm_, Acond_);
    std::cout << "  " << "Exit MINRES-QLP" << ": " << msg[istop_-1] << "\n\n";
  }
}
} // end namespace MINRESQLP
