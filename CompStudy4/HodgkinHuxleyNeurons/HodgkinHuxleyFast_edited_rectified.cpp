#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <time.h>
#include "mex.h"

#include <boost/numeric/odeint.hpp>

const double C = 1.0;
const double gNabar = 120.0;
const double gKbar = 36.0;
const double gLbar = 0.3;
const double ENa = 45.0;
const double EK = -82.0;
const double EL = -59.0;

typedef std::vector<double> state_type;

class hodgkin_huxley{
    state_type freq;
    state_type amp;
public:
    hodgkin_huxley(state_type frequency, state_type amplitude){
        freq = frequency;
        amp = amplitude;
    }
    
    double alpham (double v){
        double theta = (v+45.0)/10.0;
        return fabs(theta)<0.0001? 1.0 : 1*theta/(1.0-exp(-theta));
    }
    
    double betam(double v){
        return 1*4.0*exp(-(v+70.0)/18.0);
    }
    
    double alphah(double v){
        return 0.07*exp(-(v+70.0)/20.0);
    }
    
    double betah(double v){
        return 1.0/(1.0+exp(-(v+40.0)/10.0));
    }
    
    double alphan(double v){
        double theta = (v+60.0)/10.0;
        return fabs(theta)<0.0001? 0.1 : 0.1*theta/(1.0 - exp(-theta));
    }
    
    double betan(double v){
        return 0.125 * exp(-(v+70.0)/80.0);
    }
    
    void operator() (const state_type &s, state_type &dsdt, const double t){
        dsdt[0] = alpham(s[3]) * (1 - s[0]) - betam(s[3]) * s[0];
        dsdt[1] = alphan(s[3]) * (1 - s[1]) - betan(s[3]) * s[1];
        dsdt[2] = alphah(s[3]) * (1 - s[2]) - betah(s[3]) * s[2];
        double gNa = gNabar * pow(s[0], 3) * s[2]; //m^3 h
        double gK = gKbar * pow(s[1], 4);  // n^4
        double ix = 0;
        for (int i = 0; i < freq.size(); i++){
            ix += amp[i] * cos(2.0 * M_PI * freq[i] * t / 1000.0);
            
            
        }
        if(ix<0){ix=0;}
            
        
        dsdt[3] = - (1/C) * (gNa * (s[3]-ENa) + gK * (s[3] - EK) + gLbar * (s[3] - EL)) + ix/C;
    }
};

struct push_back_state_and_time
{
    state_type &m_states;
    state_type &m_times;
    state_type &m_m;
    state_type &m_n;
    state_type &m_h;
    
    push_back_state_and_time( state_type &states , state_type &times, state_type &m, state_type &n, state_type &h )
    : m_states( states ) , m_times( times ), m_m(m), m_n(n), m_h(h) { }
    
    void operator()( const state_type &x , double t )
    {
        m_states.push_back( x[3] );
        m_times.push_back( t );
        m_m.push_back(x[0]);
        m_n.push_back(x[1]);
        m_h.push_back(x[2]);
    }
};

void solve_HH(const double *frequency, const double *amplitude, const size_t len, const double vstart, const double max_t, const double dt, state_type &state_vec, state_type &time_vec, state_type &m_vec, state_type &n_vec, state_type &h_vec){
    using namespace boost::numeric::odeint;
    state_type freq(len), amp(len), init(4);
    for(size_t i=0; i<len; i++){
        freq[i] = frequency[i];
        amp[i] = amplitude[i];
    }
    hodgkin_huxley HHmodel(freq, amp);
    double m = HHmodel.alpham(vstart)/(HHmodel.alpham(vstart) + HHmodel.betam(vstart));
    double n = HHmodel.alphan(vstart)/(HHmodel.alphan(vstart) + HHmodel.betan(vstart));
    double h = HHmodel.alphah(vstart)/(HHmodel.alphah(vstart) + HHmodel.betah(vstart));
    init[0] = m;
    init[1] = n;
    init[2] = h;
    init[3] = vstart;
    try {
    integrate_const(make_dense_output( 1.0e-6 , 1.0e-5 , runge_kutta_dopri5< state_type >() ), HHmodel, init, 0.0, max_t, dt, push_back_state_and_time(state_vec, time_vec, m_vec, n_vec, h_vec), max_step_checker(5000));
    }
    catch (no_progress_error& e){
//        exit(1);
        for(size_t i=0; i<state_vec.size(); i++){
            state_vec[i] = -1;
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs!=5) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Five inputs required.");
    }
//     if(nlhs!=5) {
//         mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","Five output required.");
//     }
    if(mxGetN(prhs[2])!=1){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notColumnVector","freq should be a column vector");
    }
    if(mxGetN(prhs[3])!=1){
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notColumnVector","amp should be a column vector");
    }
    
    double vstart = mxGetScalar(prhs[0]);
    double max_t = mxGetScalar(prhs[1]);
    double *freq = mxGetPr(prhs[2]);
    double *amp = mxGetPr(prhs[3]);
    double dt = 1.0 / mxGetScalar(prhs[4]);
    size_t len = mxGetM(prhs[2]);
    
    state_type states;
    state_type time_vec;
    state_type m_vec;
    state_type n_vec;
    state_type h_vec;
    
    solve_HH(freq, amp, len, vstart, max_t, dt, states, time_vec, m_vec, n_vec, h_vec);
    
    mxArray *t_out = mxCreateDoubleMatrix(time_vec.size(), 1, mxREAL);
    mxArray *state_out = mxCreateDoubleMatrix(time_vec.size(), 1, mxREAL);
    mxArray *m = mxCreateDoubleMatrix(m_vec.size(), 1, mxREAL);
    mxArray *n = mxCreateDoubleMatrix(n_vec.size(), 1, mxREAL);
    mxArray *h = mxCreateDoubleMatrix(h_vec.size(), 1, mxREAL);
    
    std::copy(time_vec.begin(), time_vec.end(), mxGetPr(t_out));
    std::copy(states.begin(), states.end(), mxGetPr(state_out));
    std::copy(m_vec.begin(), m_vec.end(), mxGetPr(m));
    std::copy(n_vec.begin(), n_vec.end(), mxGetPr(n));
    std::copy(h_vec.begin(), h_vec.end(), mxGetPr(h));
    
    if(nlhs == 1){
        plhs[0] = state_out;}
    else if(nlhs == 2){
        plhs[0] = t_out;
        plhs[1] = state_out;}
    else if(nlhs == 5){    
        plhs[0] = t_out;
        plhs[1] = state_out;
        plhs[2] = m;
        plhs[3] = n;
        plhs[4] = h;}
    else
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","Number of output should be 1, 2, or 5.");
}

        