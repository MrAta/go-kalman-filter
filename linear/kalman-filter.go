package linear

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

//SysState System state including the state and covariance
type SysState struct {
	Xt *mat.VecDense // Current system state
	Pt *mat.Dense    // Current covariance matrix
}

//Noise : Measurement and System noise
type Noise struct {
	Q *mat.Dense // (discretized) system noise
	R *mat.Dense // measurement noise
}

//NewZeroNoise initializes a Noise struct
//q: dimension of square matrix Q
//r: dimension of square matrix R
func NewZeroNoise(q, r int) Noise {
	nse := Noise{
		Q: mat.NewDense(q, q, nil),
		R: mat.NewDense(r, r, nil),
	}
	return nse
}

//FilterInterface interface
type FilterInterface interface {
	Apply(sst *SysState, z, ctrl *mat.VecDense) mat.Vector
	State() mat.Vector
}

//Filter struct
type Filter struct {
	Ad         *mat.Dense
	Bd         *mat.Dense
	C          *mat.Dense
	D          *mat.Dense
	Nse        Noise
	savedState *mat.VecDense
}

//NewFilter returns a Kalman filter
func NewFilter(Ad, Bd, C, D *mat.Dense, nse Noise) FilterInterface {
	return &Filter{Ad, Bd, C, D, nse, nil}
}

//Apply implements the FilterInterface interface
func (f *Filter) Apply(sst *SysState, z, ctrl *mat.VecDense) mat.Vector {
	// correct state and covariance
	err := f.Update(sst, z, ctrl)
	if err != nil {
		fmt.Println(err)
	}

	// get response of system y

	var CXt mat.VecDense
	CXt.MulVec(f.C, sst.Xt)

	var Dctrl mat.VecDense
	Dctrl.MulVec(f.D, ctrl)

	var sum mat.VecDense
	sum.AddVec(&CXt, &Dctrl)
	filtered := &sum

	// save current context state
	f.savedState = mat.VecDenseCopyOf(sst.Xt)

	// predict new state
	f.NextState(sst, ctrl)

	// predict new covariance matrix
	f.NextCovariance(sst)

	return filtered
}

//NextState predicts next state of the system
func (f *Filter) NextState(sst *SysState, ctrl *mat.VecDense) error {
	// X_k = Ad * X_k-1 + Bd * ctrl
	//d.Ad, x, d.Bd, u
	var AdXt mat.VecDense
	AdXt.MulVec(f.Ad, sst.Xt)

	var Bdctrl mat.VecDense
	Bdctrl.MulVec(f.Bd, ctrl)

	var sum mat.VecDense
	sum.AddVec(&AdXt, &Bdctrl)
	sst.Xt = &sum
	return nil
}

//NextCovariance updates the covariance matrix
func (f *Filter) NextCovariance(sst *SysState) error {
	// P_new = Ad * P * Ad^t + Q
	//ctx.P.Product(f.Lti.Ad, ctx.P, f.Lti.Ad.T())
	//ctx.P.Add(ctx.P, f.Nse.Q)
	var pmt, mpmt mat.Dense
	pmt.Mul(sst.Pt, f.Ad.T())
	mpmt.Mul(f.Ad, &pmt)
	if f.Nse.Q != nil {
		mpmt.Add(&mpmt, f.Nse.Q)
	}
	sst.Pt = &mpmt
	return nil
}

//Update performs Kalman update
func (f *Filter) Update(sst *SysState, z, ctrl mat.Vector) error {
	// kalman gain
	// K = P C^T (C P C^T + R)^-1
	var K, kt, PCt, CPCt, denom mat.Dense
	PCt.Mul(sst.Pt, f.C.T())
	CPCt.Mul(f.C, &PCt)
	denom.Add(&CPCt, f.Nse.R)

	// calculation of Kalman gain with mat.Solve(..)
	// K = P C^T (C P C^T + R)^-1
	// K * (C P C^T + R) = P C^T
	// (C P C^T + R)^T K^T = (P C^T )^T
	err := kt.Solve(denom.T(), PCt.T())
	if err != nil {
		//log.Println(err)
		//log.Println("setting Kalman gain to zero")
		denom.Zero()
		K.Product(sst.Pt, f.C.T(), &denom)
	} else {
		K.CloneFrom(kt.T())
	}

	// update state
	// X~_k = X_k + K * [z_k - C * X_k - D * ctrl ]
	var CXk, DCtrl, bracket, Kupd mat.VecDense
	CXk.MulVec(f.C, sst.Xt)
	DCtrl.MulVec(f.D, ctrl)
	bracket.SubVec(z, &CXk)
	bracket.SubVec(&bracket, &DCtrl)
	Kupd.MulVec(&K, &bracket)
	sst.Xt.AddVec(sst.Xt, &Kupd)

	// update covariance
	// P~_k = P_k - K * [C * P_k]
	var KCP mat.Dense
	KCP.Product(&K, f.C, sst.Pt)
	sst.Pt.Sub(sst.Pt, &KCP)

	return nil
}

//State return the current state of the context
func (f *Filter) State() mat.Vector {
	var state mat.VecDense
	state.CloneVec(f.savedState)
	return &state
}
